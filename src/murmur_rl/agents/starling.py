import math

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACTION_EPS = 1e-6


def _atanh(action: torch.Tensor) -> torch.Tensor:
    action = action.clamp(min=-1.0 + ACTION_EPS, max=1.0 - ACTION_EPS)
    return 0.5 * (torch.log1p(action) - torch.log1p(-action))


def _squash_log_prob(probs: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    action = torch.tanh(raw_action)
    correction = 2.0 * (
        math.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
    )
    return (probs.log_prob(raw_action) - correction).sum(dim=-1), action


class _ActorCriticBrain(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        hidden_size: int,
        critic_hidden_size: int,
        stacked_frames: int,
    ):
        super().__init__()

        self.actor_feature_extractor = nn.Sequential(
            nn.Linear(obs_dim * stacked_frames, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Output the pre-squash mean; actions are squashed with tanh at sample/eval time.
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim * stacked_frames, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, 1),
        )

    def get_value(self, global_obs):
        return self.critic(global_obs)

    def get_deterministic_action(self, obs):
        features = self.actor_feature_extractor(obs)
        action_mean = self.actor_mean(features)
        return torch.tanh(action_mean)

    def get_action_and_value(self, obs, global_obs=None, action=None):
        features = self.actor_feature_extractor(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean).clamp(
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            raw_action = probs.rsample()
            log_prob, action = _squash_log_prob(probs, raw_action)
        else:
            raw_action = _atanh(action)
            log_prob, action = _squash_log_prob(probs, raw_action)

        # Use the base Gaussian entropy as a stable exploration proxy for PPO.
        entropy = probs.entropy().sum(dim=-1)

        value = None
        if global_obs is not None:
            value = self.get_value(global_obs)

        return action, log_prob, entropy, value


class StarlingBrain(_ActorCriticBrain):
    """
    MAPPO Actor-Critic Neural Network.
    - All starlings share the Actor network (processing local observations).
    - The Centralized Critic processes the global state to solve credit assignment.
    """

    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int = 3,
        hidden_size: int = 128,
        critic_hidden_size: int = 512,
        stacked_frames: int = 4,
    ):
        super().__init__(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            critic_hidden_size=critic_hidden_size,
            stacked_frames=stacked_frames,
        )


class FalconBrain(_ActorCriticBrain):
    """
    MAPPO Actor-Critic Neural Network for the Predator (Falcon) population.
    - Operates on the predator observation space.
    - Decoupled from the Starling weights for zero-sum Co-Evolution.
    """

    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int = 3,
        hidden_size: int = 256,
        critic_hidden_size: int = 512,
        stacked_frames: int = 4,
    ):
        super().__init__(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            critic_hidden_size=critic_hidden_size,
            stacked_frames=stacked_frames,
        )
