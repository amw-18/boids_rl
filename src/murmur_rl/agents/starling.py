import torch
import torch.nn as nn
from torch.distributions import Normal

class StarlingBrain(nn.Module):
    """
    MAPPO Actor-Critic Neural Network.
    - All starlings share the Actor network (processing local observations).
    - The Centralized Critic processes the global state to solve credit assignment.
    """
    def __init__(self, obs_dim: int, global_obs_dim: int, action_dim: int = 4, hidden_size: int = 64, critic_hidden_size: int = 256):
        super().__init__()
        
        # Actor: Shared feature extractor for local observation
        self.actor_feature_extractor = nn.Sequential(
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
        
        # Actor log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Centralized Critic: predicts global state value V(s)
        # Needs to be significantly larger to process the massive global state without bottlenecking
        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, 1)
        )

    def get_value(self, global_obs):
        """Get the value from the Centralized Critic."""
        return self.critic(global_obs)
        
    def get_action_and_value(self, obs, global_obs=None, action=None):
        """
        Produce an action from the policy (local obs) and the value (global obs).
        """
        # --- ACTOR ---
        features = self.actor_feature_extractor(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        
        # --- CRITIC ---
        value = None
        if global_obs is not None:
            value = self.get_value(global_obs)
            
        return action, log_prob, entropy, value
