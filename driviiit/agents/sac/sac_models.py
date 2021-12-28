import torch, torch.nn.functional
import numpy as np

from driviiit.agents.sac.sac_config import ConfigurationSAC


def mlp(sizes, activation=torch.nn.ReLU, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = torch.nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = torch.nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = torch.distributions.normal.Normal(mu, std)

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            tmp = 2 * (
                np.log(2) - pi_action - torch.nn.functional.softplus(-2 * pi_action)
            )
            tmp = tmp.sum(axis=(1 if len(tmp.shape) > 1 else -1))
            logp_pi -= tmp
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(torch.nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=torch.nn.ReLU,
        latent_dims=None,
        device="cpu",
    ):
        super().__init__()

        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = device
        self.to(device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy() if self.device == "cpu" else a.cpu().numpy()


def resnet18(pretrained=True):
    model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Identity()
    return model


class Qfunction(torch.nn.Module):
    """
    Modified from the core MLPQFunction and MLPActorCritic to include a speed encoder
    """

    def __init__(self):
        super().__init__()
        # pdb.set_trace()
        self.speed_encoder = mlp([1] + ConfigurationSAC.SPEED_HIDDEN_LAYER_SIZES)
        self.regressor = mlp(
            [
                ConfigurationSAC.VAE_LATENT_DIMS
                + ConfigurationSAC.SPEED_HIDDEN_LAYER_SIZES[-1]
                + 2
            ]
            + ConfigurationSAC.VISION_HIDDEN_LAYER_SIZES
            + [1]
        )

    def forward(self, obs_feat, action):
        img_embed = obs_feat[..., : ConfigurationSAC.VAE_LATENT_DIMS]  # n x latent_dims
        speed = obs_feat[..., ConfigurationSAC.VAE_LATENT_DIMS :]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        out = self.regressor(torch.cat([img_embed, spd_embed, action], dim=-1))  # n x 1
        # pdb.set_trace()
        return out.view(-1)


class DuelingNetwork(torch.nn.Module):
    """
    Further modify from Qfunction to
        - Add an action_encoder
        - Separate state-dependent value and advantage
            Q(s, a) = V(s) + A(s, a)
    """

    def __init__(self):
        super().__init__()

        self.speed_encoder = mlp([1] + ConfigurationSAC.SPEED_HIDDEN_LAYER_SIZES)
        self.action_encoder = mlp([2] + ConfigurationSAC.ACTION_HIDDEN_LAYER_SIZES)

        n_obs = (
            ConfigurationSAC.VAE_LATENT_DIMS
            + ConfigurationSAC.SPEED_HIDDEN_LAYER_SIZES[-1]
        )
        self.A_network = mlp(
            [n_obs + ConfigurationSAC.ACTION_HIDDEN_LAYER_SIZES[-1]]
            + ConfigurationSAC.VISION_HIDDEN_LAYER_SIZES
            + [1]
        )

    def forward(self, obs_feat, action, advantage_only=False):
        img_embed = obs_feat[..., : ConfigurationSAC.VAE_LATENT_DIMS]  # n x latent_dims
        speed = obs_feat[..., ConfigurationSAC.VAE_LATENT_DIMS :]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        action_embed = self.action_encoder(action)

        out = self.A_network(torch.cat([img_embed, spd_embed, action_embed], dim=-1))
        """
        if advantage_only == False:
            V = self.V_network(torch.cat([img_embed, spd_embed], dim = -1)) # n x 1
            out += V
        """
        return out.view(-1)


class ActorCritic(torch.nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        activation=torch.nn.ReLU,
        latent_dims=None,
        device="cpu",
        safety=False,  # Flag to indicate architecture for Safety_actor_critic
    ):
        super().__init__()
        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.speed_encoder = mlp([1] + ConfigurationSAC.SPEED_HIDDEN_LAYER_SIZES)
        self.policy = SquashedGaussianMLPActor(
            obs_dim,
            act_dim,
            ConfigurationSAC.ACTOR_HIDDEN_LAYER_SIZES,
            activation,
            act_limit,
        )
        if safety:
            self.q1 = DuelingNetwork()
        else:
            self.q1 = Qfunction()
            self.q2 = Qfunction()
        self.device = device
        self.to(device)

    def pi(self, obs_feat, deterministic=False):
        img_embed = obs_feat[..., : ConfigurationSAC.VAE_LATENT_DIMS]  # n x latent_dims
        speed = obs_feat[..., ConfigurationSAC.VAE_LATENT_DIMS :]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 8
        feat = torch.cat([img_embed, spd_embed], dim=-1)
        return self.policy(feat, deterministic, True)

    def act(self, obs_feat, deterministic=False):
        with torch.no_grad():
            img_embed = obs_feat[
                ..., : ConfigurationSAC.VAE_LATENT_DIMS
            ]  # n x latent_dims
            speed = obs_feat[..., ConfigurationSAC.VAE_LATENT_DIMS :]  # n x 1
            spd_embed = self.speed_encoder(speed)  # n x 8
            feat = torch.cat([img_embed, spd_embed], dim=-1)
            a, _ = self.policy(feat, deterministic, False)
            a = a.squeeze(0)
        return a.numpy() if self.device == "cpu" else a.cpu().numpy()
