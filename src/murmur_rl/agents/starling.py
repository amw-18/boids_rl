import torch
import torch.nn as nn
from torch.distributions import Normal

class StarlingBrain(nn.Module):
    """
    Actor-Critic Neural Network for a single Starling (Boid).
    For MAPPO, all starlings will share this same brain (parameter sharing),
    processing their localized observations to choose a steering force.
    """
    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_size: int = 64):
        super().__init__()
        
        # Shared feature extractor (optional, but good for processing neighbors)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Actor: outputs mean of the action distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() # Actions bounded [-1, 1] mapped to max_force in env
        )
        
        # Actor log standard deviation (trainable parameter independent of state)
        # Initialize to 0.0 (std = 1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic: predicts state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        features = self.feature_extractor(obs)
        
        # Actor
        action_mean = self.actor_mean(features)
        # Expand logstd to batch size
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value
        
    def get_action_and_value(self, obs, action=None):
        """
        Produce an action from the policy and the value of the observation.
        If action is provided, calculate the log probability of that action.
        """
        action_mean, action_std, value = self(obs)
        
        # Create normal distribution
        probs = Normal(action_mean, action_std)
        
        if action is None:
            # Sample action
            action = probs.sample()
            
        # Log prob of the action
        # Actions are squashed between [-1, 1] by Tanh in the network, so we don't strictly need 
        # a squashed distribution like in SAC unless we want to, but PPO usually handles boundaries
        # by simply clipping. Here, our mean is Tanh bounded, but the sampled action can exceed [-1, 1],
        # so we'll clip it when interacting with the environment, or use proper squashing.
        # For a simple baseline, standard PPO Normal clipping is fine.
        
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
