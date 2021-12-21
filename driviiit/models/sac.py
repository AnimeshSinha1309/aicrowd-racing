import numpy as np
import torch, torch.nn.functional

import ipdb as pdb


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation=torch.nn.ReLU, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


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
        try:
            pi_distribution = torch.distributions.normal.Normal(mu, std)
        except ValueError:
            pdb.set_trace()
            pass

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
            tmp = (2 * (np.log(2) - pi_action - torch.nn.functional.softplus(-2 * pi_action)))
            tmp = tmp.sum(axis=(1 if len(tmp.shape) > 1 else -1))
            logp_pi -= tmp
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class Qfunction(torch.nn.Module):
    '''
    Modified from the core MLPQFunction and MLPActorCritic to include a speed encoder
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # pdb.set_trace()
        self.speed_encoder = mlp([1] + self.cfg[self.cfg['use_encoder_type']]['speed_hiddens'])
        self.regressor = mlp([self.cfg[self.cfg['use_encoder_type']]['latent_dims'] +
                              self.cfg[self.cfg['use_encoder_type']]['speed_hiddens'][-1] + 2] +
                             self.cfg[self.cfg['use_encoder_type']]['hiddens'] + [1])
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat, action):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[..., :self.cfg[self.cfg['use_encoder_type']]['latent_dims']]  # n x latent_dims
        speed = obs_feat[..., self.cfg[self.cfg['use_encoder_type']]['latent_dims']:]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        out = self.regressor(torch.cat([img_embed, spd_embed, action], dim=-1))  # n x 1
        # pdb.set_trace()
        return out.view(-1)


class DuelingNetwork(torch.nn.Module):
    '''
    Further modify from Qfunction to
        - Add an action_encoder
        - Separate state-dependent value and advantage
            Q(s, a) = V(s) + A(s, a)
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.speed_encoder = mlp([1] + self.cfg[self.cfg['use_encoder_type']]['speed_hiddens'])
        self.action_encoder = mlp([2] + self.cfg[self.cfg['use_encoder_type']]['action_hiddens'])

        n_obs = self.cfg[self.cfg['use_encoder_type']]['latent_dims'] + \
                self.cfg[self.cfg['use_encoder_type']]['speed_hiddens'][-1]
        # self.V_network = mlp([n_obs] + self.cfg[self.cfg['use_encoder_type']]['hiddens'] + [1])
        self.A_network = mlp([n_obs + self.cfg[self.cfg['use_encoder_type']]['action_hiddens'][-1]] +
                             self.cfg[self.cfg['use_encoder_type']]['hiddens'] + [1])
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat, action, advantage_only=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[..., :self.cfg[self.cfg['use_encoder_type']]['latent_dims']]  # n x latent_dims
        speed = obs_feat[..., self.cfg[self.cfg['use_encoder_type']]['latent_dims']:]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        action_embed = self.action_encoder(action)

        out = self.A_network(torch.cat([img_embed, spd_embed, action_embed], dim=-1))
        '''
        if advantage_only == False:
            V = self.V_network(torch.cat([img_embed, spd_embed], dim = -1)) # n x 1
            out += V
        '''
        return out.view(-1)


class ActorCritic(torch.nn.Module):
    def __init__(self, observation_space, action_space, cfg,
                 activation=torch.nn.ReLU, latent_dims=None, device='cpu',
                 safety=False  ## Flag to indicate architecture for Safety_actor_critic
                 ):
        super().__init__()
        self.cfg = cfg
        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.speed_encoder = mlp([1] + self.cfg[self.cfg['use_encoder_type']]['speed_hiddens'])
        self.policy = SquashedGaussianMLPActor(
            obs_dim, act_dim, cfg[cfg['use_encoder_type']]['actor_hiddens'], activation, act_limit)
        if safety:
            self.q1 = DuelingNetwork(cfg)
        else:
            self.q1 = Qfunction(cfg)
            self.q2 = Qfunction(cfg)
        self.device = device
        self.to(device)

    def pi(self, obs_feat, deterministic=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[..., :self.cfg[self.cfg['use_encoder_type']]['latent_dims']]  # n x latent_dims
        speed = obs_feat[..., self.cfg[self.cfg['use_encoder_type']]['latent_dims']:]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 8
        feat = torch.cat([img_embed, spd_embed], dim=-1)
        return self.policy(feat, deterministic, True)

    def act(self, obs_feat, deterministic=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        with torch.no_grad():
            img_embed = obs_feat[..., :self.cfg[self.cfg['use_encoder_type']]['latent_dims']]  # n x latent_dims
            speed = obs_feat[..., self.cfg[self.cfg['use_encoder_type']]['latent_dims']:]  # n x 1
            # pdb.set_trace()
            spd_embed = self.speed_encoder(speed)  # n x 8
            feat = torch.cat([img_embed, spd_embed], dim=-1)
            a, _ = self.policy(feat, deterministic, False)
            a = a.squeeze(0)
        return a.numpy() if self.device == 'cpu' else a.cpu().numpy()
