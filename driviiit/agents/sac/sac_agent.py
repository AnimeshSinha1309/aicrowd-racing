"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
import queue, threading
from copy import deepcopy

import torch
import numpy as np
import wandb
from gym.spaces import Box
from torch.optim import Adam

from driviiit.interface.metas import BaseAgent
from driviiit.agents.sac.sac_models import ActorCritic
from driviiit.models.vae import VAE
from driviiit.loggers.replay_buffer import ReplayBuffer
from driviiit.agents.sac.sac_config import ConfigurationSAC
from l2r.common.utils import RecordExperience
from l2r.common.utils import setup_logging


DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class SACAgent(BaseAgent):
    """Adopted from https://github.com/learn-to-race/l2r/blob/main/l2r/baselines/rl/sac.py"""

    def __init__(self):
        super(SACAgent, self).__init__()

        self.file_logger, _ = self.setup_loggers()

        if ConfigurationSAC.record_experience:
            self.save_queue = queue.Queue()
            self.save_batch_size = 256
            self.record_experience = RecordExperience(
                ConfigurationSAC.record_dir,
                ConfigurationSAC.track_name,
                ConfigurationSAC.experiment_name,
                self.file_logger,
                self,
            )
            self.save_thread = threading.Thread(
                target=self.record_experience.save_thread
            )
            self.save_thread.start()

        # Setup Vision Encoder
        self.backbone = VAE(z_dim=ConfigurationSAC.VAE_LATENT_DIMS)
        self.backbone.load_state_dict(
            torch.load("data/models/vae_rgb_front.pth", map_location=DEVICE)
        )
        self.backbone.to(DEVICE)

        self.feat_dim = ConfigurationSAC.VAE_LATENT_DIMS + 1
        self.obs_dim = (
            ConfigurationSAC.VAE_LATENT_DIMS
            + ConfigurationSAC.SPEED_HIDDEN_LAYER_SIZES[-1]
        )

        self.set_params()

    def select_action(self, obs, encode=True):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if encode:
            obs = self.encode_state(obs)
        if self.t > ConfigurationSAC.START_STEPS:
            a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
            a = a  # numpy array...
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return a

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        # camera, features, state = obs
        self.deterministic = True
        self.t = 1e6

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def set_params(self):
        self.save_episodes = True
        self.episode_num = 0
        self.best_ret = 0
        self.t = 0
        self.deterministic = False
        self.atol = 1e-3
        self.store_from_safe = False
        self.pi_scheduler = None
        self.t_start = 0
        self.best_pct = 0

        # This is important: it allows child classes (that extend this one) to "push up" information
        # that this parent class should log
        self.metadata = {}
        self.record = {"transition_actor": ""}

        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.feat_dim,
            act_dim=self.act_dim,
            size=ConfigurationSAC.REPLAY_MAX_STORAGE,
        )

        self.actor_critic = ActorCritic(
            self.obs_dim,
            self.action_space,
            latent_dims=self.obs_dim,
            device=DEVICE,
        )

        if ConfigurationSAC.checkpoint and ConfigurationSAC.load_checkpoint:
            self.load_model(ConfigurationSAC.checkpoint)

        self.actor_critic_target = deepcopy(self.actor_critic)

    def setup_loggers(self):
        save_path = ConfigurationSAC.model_save_path
        loggers = setup_logging(save_path, ConfigurationSAC.experiment_name, True)
        return loggers

    def compute_loss_q(self, data):
        """Set up function for computing SAC Q-losses."""
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critic_target.q1(o2, a2)
            q2_pi_targ = self.actor_critic_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + ConfigurationSAC.GAMMA * (1 - d) * (
                q_pi_targ - ConfigurationSAC.ALPHA * logp_a2
            )

        # MSE loss against Bellman backup
        loss_q1 = (self.replay_buffer.weights * (q1 - backup) ** 2).mean()
        loss_q2 = (self.replay_buffer.weights * (q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy()
        )

        return loss_q, q_info

    def compute_loss_pi(self, data):
        """Set up function for computing SAC pi loss."""
        o = data["obs"]
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (ConfigurationSAC.ALPHA * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.actor_critic.parameters(), self.actor_critic_target.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(ConfigurationSAC.POLYAK)
                p_targ.data.add_((1 - ConfigurationSAC.POLYAK) * p.data)

    def _step(self, env, action):
        obs, reward, done, info = env.step(action)
        return obs[1], self.encode_state(obs), obs[0], reward, done, info

    def _reset(self, env, random_pos=False):
        camera = 0
        while (np.mean(camera) == 0) | (np.mean(camera) == 255):
            obs = env.reset(random_pos=random_pos)
            (state, camera), _ = obs
        return camera, self.encode_state((state, camera)), state

    def encode_state(self, o):
        state, img = o

        img_embed = self.backbone.encode_raw(np.array(img), DEVICE)[0][0]
        speed = (
            torch.tensor((state[4] ** 2 + state[3] ** 2 + state[5] ** 2) ** 0.5)
            .float()
            .reshape(1, -1)
            .to(DEVICE)
        )
        out = torch.cat([img_embed.unsqueeze(0), speed], dim=-1).squeeze(
            0
        )  # torch.Size([33])

        assert not torch.sum(torch.isnan(out)), "found a nan value"
        out[torch.isnan(out)] = 0

        return out

    def eval(self, n_eps, env):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        assert ConfigurationSAC.num_test_episodes == 1

        for j in range(ConfigurationSAC.num_test_episodes):
            camera, features, state = self._reset(env, random_pos=False)
            episode_done, ep_ret, ep_len, n_val_steps, self.metadata = (
                False,
                0,
                0,
                0,
                {},
            )
            camera, features, state_next, r, episode_done, info = self._step(
                env, [0, 1]
            )
            experience, time = [], 0

            while (not episode_done) & (ep_len <= ConfigurationSAC.MAX_EPISODE_LENGTH):
                # Take deterministic actions at test time
                self.deterministic = True
                self.t = 1e6
                a = self.select_action(features, encode=False)
                (
                    camera_next,
                    features_next,
                    state_next,
                    r,
                    episode_done,
                    info,
                ) = self._step(env, a)

                # Check that the camera is turned on
                assert (np.mean(camera_next) > 0) & (np.mean(camera_next) < 255)

                ep_ret += r
                ep_len += 1
                n_val_steps += 1

                # Prevent the agent from being stuck
                if np.allclose(state_next[15:16], state[15:16], atol=self.atol, rtol=0):
                    # self.file_logger("Sampling random action to get unstuck")
                    a = env.action_space.sample()
                    # Step the env
                    (
                        camera_next,
                        features_next,
                        state_next,
                        r,
                        episode_done,
                        info,
                    ) = self._step(env, a)
                    ep_len += 1

                if ConfigurationSAC.record_experience:
                    recording = self.add_experience(
                        action=a,
                        camera=camera,
                        next_camera=camera_next,
                        done=episode_done,
                        env=env,
                        feature=features,
                        next_feature=features_next,
                        info=info,
                        state=state,
                        next_state=state_next,
                        step=time,
                        reward=r,
                    )
                    experience.append(recording)

                features = features_next
                camera = camera_next
                state = state_next
                time += 1

            self.file_logger(f"[eval episode] {info}")

            val_ep_rets.append(ep_ret)
            self.metadata["info"] = info
            wandb.log(
                {
                    "global_step": n_eps,
                    "val/episodic_return": ep_ret,
                    "val/ep_n_steps": n_val_steps,
                    "val/ep_pct_complete": info["metrics"]["pct_complete"],
                    "val/ep_total_time": info["metrics"]["total_time"],
                    "val/ep_total_distance": info["metrics"]["total_distance"],
                    "val/ep_avg_speed": info["metrics"]["average_speed_kph"],
                    "val/ep_avg_disp_err": info["metrics"][
                        "average_displacement_error"
                    ],
                    "val/ep_traj_efficiency": info["metrics"]["trajectory_efficiency"],
                    "val/ep_traj_admissibility": info["metrics"][
                        "trajectory_admissibility"
                    ],
                    "val/movement_smoothness": info["metrics"]["movement_smoothness"],
                }
            )

            # Quickly dump recently-completed episode's experience to the multithread queue,
            # as long as the episode resulted in "success"
            if (
                ConfigurationSAC.record_experience
            ):  # and self.metadata['info']['success']:
                self.file_logger("writing experience")
                self.save_queue.put(experience)

        self.checkpoint_model(ep_ret, n_eps)
        self.update_best_pct_complete(info)

        return val_ep_rets

    def update_best_pct_complete(self, info):
        if self.best_pct < info["metrics"]["pct_complete"]:
            for cutoff in [93, 100]:
                if (self.best_pct < cutoff) & (
                    info["metrics"]["pct_complete"] >= cutoff
                ):
                    self.pi_scheduler.step()
            self.best_pct = info["metrics"]["pct_complete"]

    def checkpoint_model(self, ep_ret, n_eps):
        # Save if best (or periodically)
        if ep_ret > self.best_ret:  # and ep_ret > 100):
            path_name = f"{ConfigurationSAC.model_save_path}/best_{ConfigurationSAC.experiment_name}_episode_{n_eps}.statedict"
            self.file_logger(
                f"New best episode reward of {round(ep_ret, 1)}! Saving: {path_name}"
            )
            self.best_ret = ep_ret
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{ConfigurationSAC.model_save_path}/best_{ConfigurationSAC.experiment_name}_episode_{n_eps}.statedict"

        elif self.save_episodes and (n_eps + 1 % ConfigurationSAC.save_freq == 0):
            path_name = f"{ConfigurationSAC.model_save_path}/{ConfigurationSAC.experiment_name}_episode_{n_eps}.statedict"
            self.file_logger(
                f"Periodic save (save_freq of {ConfigurationSAC.save_freq}) to {path_name}"
            )
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{ConfigurationSAC.model_save_path}/{ConfigurationSAC.experiment_name}_episode_{n_eps}.statedict"

    def training(self, env):
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(
            self.actor_critic.policy.parameters(), lr=ConfigurationSAC.LEARNING_RATE
        )
        self.q_optimizer = Adam(self.q_params, lr=ConfigurationSAC.LEARNING_RATE)
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

        # Prepare for interaction with environment
        # start_time = time.time()
        best_ret, ep_ret, ep_len = 0, 0, 0

        self._reset(env, random_pos=True)
        camera, feat, state, r, d, info = self._step(env, [0, 1])

        experience = []
        speed_dim = 1
        assert (
            len(feat) == ConfigurationSAC.VAE_LATENT_DIMS + speed_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = self.t_start
        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.t_start, ConfigurationSAC.total_steps):
            a = self.select_action(feat, encode=False)

            # Step the env
            camera2, feat2, state2, r, d, info = self._step(env, a)

            # Check that the camera is turned on
            assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

            # Prevents the agent from getting stuck by sampling random actions
            # self.atol for SafeRandom and SPAR are set to -1 so that this condition does not activate
            if np.allclose(state2[15:16], state[15:16], atol=self.atol, rtol=0):
                # self.file_logger("Sampling random action to get unstuck")
                a = env.action_space.sample()

                # Step the env
                camera2, feat2, state2, r, d, info = self._step(env, a)
                ep_len += 1

            state = state2
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == ConfigurationSAC.MAX_EPISODE_LENGTH else d

            # Store experience to replay buffer
            if (not np.allclose(state2[15:16], state[15:16], atol=3e-1, rtol=0)) | (
                r != 0
            ):
                self.replay_buffer.store(feat, a, r, feat2, d)
            else:
                # print('Skip')
                skip = True

            if ConfigurationSAC.record_experience:
                recording = self.add_experience(
                    action=a,
                    camera=camera,
                    next_camera=camera2,
                    done=d,
                    env=env,
                    feature=feat,
                    next_feature=feat2,
                    info=info,
                    reward=r,
                    state=state,
                    next_state=state2,
                    step=t,
                )
                experience.append(recording)

                # quickly pass data to save thread
                # if len(experience) == self.save_batch_size:
                #    self.save_queue.put(experience)
                #    experience = []

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            feat = feat2
            state = state2  # in case we, later, wish to store the state in the replay as well
            camera = camera2  # in case we, later, wish to store the state in the replay as well

            # Update handling
            if (t >= ConfigurationSAC.UPDATE_AFTER) & (
                t % ConfigurationSAC.UPDATE_EVERY_STEPS == 0
            ):
                for j in range(ConfigurationSAC.UPDATE_EVERY_STEPS):
                    batch = self.replay_buffer.sample_batch(
                        ConfigurationSAC.REPLAY_BATCH_SIZE
                    )
                    self.update(data=batch)

            if (t + 1) % ConfigurationSAC.EVAL_EVERY_STEPS == 0:
                # eval on test environment
                val_returns = self.eval(t // ConfigurationSAC.EVAL_EVERY_STEPS, env)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

            # End of trajectory handling
            if d or (ep_len == ConfigurationSAC.MAX_EPISODE_LENGTH):
                self.metadata["info"] = info
                self.episode_num += 1
                msg = f"[Ep {self.episode_num }] {self.metadata}"
                self.file_logger(msg)
                wandb.log(
                    {
                        "global_step": self.episode_num,
                        "train/episodic_return": ep_ret,
                        "train/ep_total_time": self.metadata["info"]["metrics"][
                            "total_time"
                        ],
                        "train/ep_total_distance": self.metadata["info"]["metrics"][
                            "total_distance"
                        ],
                        "train/ep_avg_speed": self.metadata["info"]["metrics"][
                            "average_speed_kph"
                        ],
                        "train/ep_avg_disp_err": self.metadata["info"]["metrics"][
                            "average_displacement_error"
                        ],
                        "train/ep_traj_efficiency": self.metadata["info"]["metrics"][
                            "trajectory_efficiency"
                        ],
                        "train/ep_traj_admissibility": self.metadata["info"]["metrics"][
                            "trajectory_admissibility"
                        ],
                        "train/movement_smoothness": self.metadata["info"]["metrics"][
                            "movement_smoothness"
                        ],
                        "train/ep_n_steps": t - t_start,
                    }
                )

                # Quickly dump recently-completed episode's experience to the multithread queue,
                # as long as the episode resulted in "success"
                if (
                    ConfigurationSAC.record_experience
                ):  # and self.metadata['info']['success']:
                    self.file_logger("Writing experience")
                    self.save_queue.put(experience)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_episode(env, t)

    def reset_episode(self, env, t):
        camera, feat, state = self._reset(env, random_pos=True)
        ep_ret, ep_len, self.metadata, experience = 0, 0, {}, []
        t_start = t + 1
        camera, feat, state2, r, d, info = self._step(env, [0, 1])
        return camera, ep_len, ep_ret, experience, feat, state, t_start

    def add_experience(
        self,
        action,
        camera,
        next_camera,
        done,
        env,
        feature,
        next_feature,
        info,
        reward,
        state,
        next_state,
        step,
    ):
        self.recording = {
            "step": step,
            "nearest_idx": env.nearest_idx,
            "camera": camera,
            "feature": feature.detach().cpu().numpy(),
            "state": state,
            "action_taken": action,
            "next_camera": next_camera,
            "next_feature": next_feature.detach().cpu().numpy(),
            "next_state": next_state,
            "reward": reward,
            "episode": self.episode_num,
            "stage": "training",
            "done": done,
            "transition_actor": self.record["transition_actor"],
            "metadata": info,
        }
        return self.recording
