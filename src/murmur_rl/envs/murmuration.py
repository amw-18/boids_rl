import functools
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
import torch
import numpy as np

from murmur_rl.envs.physics import BoidsPhysics

class MurmurationEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for the Murmuration Simulation.
    Each starling is an agent controlled by a neural network.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "murmuration_v0"
    }

    def __init__(
        self,
        num_agents=50,
        space_size=100.0,
        perception_radius=10.0,
        render_mode=None,
        device="cpu"
    ):
        self.n_agents = num_agents
        self.space_size = space_size
        self.perception_radius = perception_radius
        self.device = torch.device(device)
        self.render_mode = render_mode
        
        self.possible_agents = [f"starling_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # Initialize the underlying PyTorch physics engine
        # We disable normal rules in physics engine if we want RL to learn them entirely,
        # but for murmuration we might want to start with normal boids and RL learns to tweak them, 
        # or RL learns everything from scratch.
        # Let's say RL learns to supply the raw steering force (X, Y, Z).
        self.physics = BoidsPhysics(
            num_boids=num_agents,
            space_size=space_size,
            device=self.device,
            perception_radius=perception_radius
        )

        # Action space: 3D continuous force vector [-1, 1] mapped to max_force
        # Observation space (16 dimensions):
        # - Agent's own velocity (3)
        # - Group Context:
        #   - nearest_dist (1)
        #   - local_density (1)
        #   - local_alignment (3)
        #   - com_direction (3)
        # - Perceptual Threat (Predator):
        #   - distance (1)
        #   - v_close (1)
        #   - loom (1)
        #   - in_front (1)
        # - Boundary distances (1) - distance to closest wall
        
        obs_dim = 3 + 1 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 1
        
        # Spaces are defined in NumPy for PettingZoo standard compliance,
        # even though computation is in PyTorch.
        self.action_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for agent in self.possible_agents
        }
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.physics.reset() # Reset positions and velocities on GPU
        
        # Generate initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        actions: dict of agent_name -> numpy array (3,)
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Convert dictionary of actions to PyTorch tensor
        # Assumes actions keys are ordered same as self.agents
        action_list = [actions[agent] for agent in self.possible_agents]
        action_tensor = torch.tensor(np.array(action_list), dtype=torch.float32, device=self.device)
        
        # Scale actions to max force
        # Assuming physics max_force internally limits it, we just pass the vector
        self.physics.step(actions=action_tensor)
        
        self.num_moves += 1

        # Calculate rewards and observations
        observations = self._get_observations()
        rewards = self._get_rewards()
        
        # Truncate after 500 steps
        env_truncation = self.num_moves >= 500
        truncations = {agent: env_truncation for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        """Build biological observations purely in PyTorch, then convert to numpy dict."""
        obs_dict = {}
        
        pos = self.physics.positions  # (N, 3)
        vel = self.physics.velocities # (N, 3)
        alive = self.physics.alive_mask # (N,)
        
        # Precompute pairwise distances among ALIVE boids
        # To keep dimensions aligned, we compute for all, but mask dead
        dist_matrix = torch.cdist(pos, pos) # (N, N)
        dist_matrix.fill_diagonal_(float('inf'))
        dist_matrix.masked_fill_(~alive.unsqueeze(0), float('inf'))
        
        # === Group Context ===
        # 1. Nearest Dist
        nearest_dist, _ = torch.min(dist_matrix, dim=1, keepdim=True) # (N, 1)
        nearest_dist.masked_fill_(nearest_dist == float('inf'), self.perception_radius) # If alone, assume far
        
        # 2. Local Density (num within radius)
        in_radius_mask = dist_matrix < self.perception_radius # (N, N)
        local_density = in_radius_mask.sum(dim=1, keepdim=True).float() # (N, 1)
        local_density = local_density / self.n_agents # Normalize 0-1
        
        # 3. Local Alignment & COM Direction
        local_alignment = torch.zeros_like(vel)
        com_direction = torch.zeros_like(pos)
        
        for i in range(self.n_agents):
            if not alive[i]:
                continue
            neighbors_mask = in_radius_mask[i]
            if neighbors_mask.any():
                # Alignment
                neighbor_vels = vel[neighbors_mask]
                avg_vel = neighbor_vels.mean(dim=0)
                norm_avg_vel = avg_vel / (torch.norm(avg_vel).clamp(min=1e-5))
                # Map to self heading (dot product is simple, but returning vectors is fine too)
                local_alignment[i] = norm_avg_vel
                
                # COM Direction
                neighbor_pos = pos[neighbors_mask]
                com = neighbor_pos.mean(dim=0)
                dir_to_com = com - pos[i]
                norm_dir = dir_to_com / (torch.norm(dir_to_com).clamp(min=1e-5))
                com_direction[i] = norm_dir
                
        # === Perceptual Threat (Predator) ===
        predator_pos = self.physics.predator_position # (1, 3)
        predator_vel = self.physics.predator_velocity # (1, 3)
        
        dx = predator_pos - pos # (N, 3)
        dv = predator_vel - vel # (N, 3)
        
        # 1. Distance d
        d = torch.norm(dx, dim=-1, keepdim=True) # (N, 1)
        
        # 2. Unit direction
        u = dx / d.clamp(min=1e-5) # (N, 3)
        
        # 3. Closing Speed (v_close)
        # -dot(dv, u)
        v_close = -torch.sum(dv * u, dim=-1, keepdim=True) # (N, 1)
        
        # 4. Looming (Time-to-collision proxy)
        loom = v_close / d.clamp(min=1e-5) # (N, 1)
        
        # 5. Bearing (in_front)
        # dot(self_vel_unit, u)
        self_vel_unit = vel / (torch.norm(vel, dim=-1, keepdim=True).clamp(min=1e-5))
        in_front = torch.sum(self_vel_unit * u, dim=-1, keepdim=True) # (N, 1)
        
        # Mask threats if far away (visual radius)
        threat_mask = d > (self.space_size / 2.0)
        v_close.masked_fill_(threat_mask, 0.0)
        loom.masked_fill_(threat_mask, 0.0)
        in_front.masked_fill_(threat_mask, 0.0)
        
        # === Bounds ===
        # Distance to closest wall
        dist_to_bounds_x = torch.min(pos[:, 0], self.space_size - pos[:, 0]).unsqueeze(1)
        dist_to_bounds_y = torch.min(pos[:, 1], self.space_size - pos[:, 1]).unsqueeze(1)
        dist_to_bounds_z = torch.min(pos[:, 2], self.space_size - pos[:, 2]).unsqueeze(1)
        closest_wall = torch.min(torch.cat([dist_to_bounds_x, dist_to_bounds_y, dist_to_bounds_z], dim=1), dim=1, keepdim=True)[0]
        
        # Package for agents
        for i, agent in enumerate(self.possible_agents):
            if not alive[i]:
                # Agent is dead, PettingZoo expects us to remove it from `self.agents` 
                # and NOT return observation for it next step. Handled in step().
                if agent in self.agents:
                    self.agents.remove(agent)
                continue
                
            obs = torch.cat([
                vel[i],                    # 3
                nearest_dist[i],           # 1
                local_density[i],          # 1
                local_alignment[i],        # 3
                com_direction[i],          # 3
                d[i],                      # 1
                v_close[i],                # 1
                loom[i],                   # 1
                in_front[i],               # 1
                closest_wall[i]            # 1
            ]).cpu().numpy()
            
            obs_dict[agent] = obs
            
        return obs_dict

    def _get_rewards(self):
        """
        Biological Phase 4 Reward:
        1. Survival Reward (+)
        2. Death Penalty (-) (Predator catch or wall hit)
        3. Social Proximity Reward (+) 
        4. Collision Penalty (-)
        5. Energy penalty (-) (assuming NN outputs force mapped later, currently implicit)
        """
        rewards = {}
        pos = self.physics.positions
        alive = self.physics.alive_mask
        
        # Pairwise distance
        dist_matrix = torch.cdist(pos, pos)
        dist_matrix.fill_diagonal_(float('inf'))
        
        nearest_dist, _ = torch.min(dist_matrix, dim=1)
        
        # Social: how many neighbors in comfortable radius (e.g., between 2.0 and 10.0)
        comfortable_mask = (dist_matrix > 2.0) & (dist_matrix <= self.perception_radius)
        social_count = comfortable_mask.sum(dim=1).float()
        
        # Collision: distance < 1.0 is bad
        collision_count = (dist_matrix < 1.0).sum(dim=1).float()
        
        # Bounds check for death
        margin = 1.0
        hit_wall = (pos < margin).any(dim=1) | (pos > (self.space_size - margin)).any(dim=1)
        
        for i, agent in enumerate(self.possible_agents):
            # Only give reward if agent was alive in step
            if agent not in self.agents:
                 continue
            
            # If agent just died (not alive now, but was in self.agents)
            if not alive[i]:
                # Death by predator
                rewards[agent] = -100.0
                continue
                
            if hit_wall[i]:
                # Death by wall
                rewards[agent] = -100.0
                # Technically should kill them here, but env truncation will handle it 
                # or next step will freeze them.
                continue
                
            # Stay alive base reward
            rew = 0.1 
            
            # Social reward
            if social_count[i] > 0:
                 rew += 0.05 * min(social_count[i].item(), 5.0) # Cap at 5 neighbors worth
                 
            # Collision penalty
            if collision_count[i] > 0:
                 rew -= 2.0 * collision_count[i].item()
                 
            rewards[agent] = rew
             
        return rewards

    def render(self):
        pass

    def close(self):
        pass
