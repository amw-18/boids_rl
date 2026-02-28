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

    def __init__(self, num_agents=50, num_predators=5, space_size=100.0, perception_radius=10.0, device='cpu', gamma=0.99, pbrs_k=1.0, pbrs_c=1.0):
        super().__init__()
        self.n_agents = num_agents
        self.num_predators = num_predators
        self.space_size = space_size
        self.perception_radius = perception_radius
        self.device = torch.device(device)
        self.gamma = gamma
        self.pbrs_k = pbrs_k
        self.pbrs_c = pbrs_c
        self.last_potential = {}
        
        self.possible_agents = [f"boid_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # Instantiate Vectorized Physics Engine
        # We disable normal rules in physics engine if we want RL to learn them entirely,
        # but for murmuration we might want to start with normal boids and RL learns to tweak them, 
        # or RL learns everything from scratch.
        # Let's say RL learns to supply the raw steering force (X, Y, Z).
        self.physics = BoidsPhysics(
            num_boids=num_agents,
            num_predators=num_predators,
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
        obs_dim = 3 + 1 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 3
        
        # Spaces are defined in NumPy for PettingZoo standard compliance,
        # even though computation is in PyTorch.
        # Action space: 3D continuous vector
        # [0] -> Thrust, [1] -> Roll, [2] -> Pitch
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
        self.dead_agents = set() # Track who is newly dead
        
        # Generate initial potentials
        _, initial_potentials = self._get_rewards()
        self.last_potential = initial_potentials
        
        # Generate initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.possible_agents}
        
        return observations, infos

    def step(self, actions):
        """
        actions: dict of agent_name -> numpy array (3,)
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Convert dictionary of actions to PyTorch tensor
        # Map over possible_agents directly to preserve shape N
        # Actions are 3D
        action_list = [actions.get(agent, np.zeros(3)) for agent in self.possible_agents]
        action_tensor = torch.tensor(np.array(action_list), dtype=torch.float32, device=self.device)
        
        self.physics.step(boid_actions=action_tensor)
        
        self.num_moves += 1

        # Calculate rewards and observations
        observations = self._get_observations()
        base_rewards, new_potentials = self._get_rewards()
        
        rewards = {}
        for agent in self.possible_agents:
            if agent in self.dead_agents and agent not in base_rewards:
                rewards[agent] = base_rewards.get(agent, 0.0)
            else:
                shaping = (self.gamma * new_potentials[agent]) - self.last_potential.get(agent, 0.0)
                rewards[agent] = base_rewards[agent] + shaping
            self.last_potential[agent] = new_potentials[agent]
        
        # Truncate after 500 steps
        env_truncation = self.num_moves >= 500
        truncations = {agent: env_truncation for agent in self.possible_agents}
        terminations = {agent: (agent in self.dead_agents) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

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
        # 1. Nearest Dist (Normalized 0-1)
        nearest_dist, _ = torch.min(dist_matrix, dim=1, keepdim=True) # (N, 1)
        nearest_dist.masked_fill_(nearest_dist == float('inf'), self.perception_radius) # If alone, assume far
        nearest_dist = nearest_dist / self.perception_radius
        
        # 2. Local Density (num within radius) (Already Normalized 0-1)
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
        predator_pos = self.physics.predator_position # (num_predators, 3)
        predator_vel = self.physics.predator_velocity # (num_predators, 3)
        
        # Find closest predator for each boid
        dist_to_preds = torch.cdist(pos, predator_pos) # (N, num_predators)
        closest_pred_idx = torch.argmin(dist_to_preds, dim=1) # (N,)
        
        closest_pred_pos = predator_pos[closest_pred_idx] # (N, 3)
        closest_pred_vel = predator_vel[closest_pred_idx] # (N, 3)
        
        dx = closest_pred_pos - pos # (N, 3)
        dv = closest_pred_vel - vel # (N, 3)
        
        # 1. Distance d (Normalized 0-1 based on half space size)
        d = torch.norm(dx, dim=-1, keepdim=True) # (N, 1)
        d_norm = torch.clamp(d / (self.space_size / 2.0), max=1.0)
        
        # 2. Unit direction
        u = dx / d.clamp(min=1e-5) # (N, 3)
        
        # 3. Closing Speed (v_close) (Normalized -1 to 1)
        # Max closing speed is roughly predator_speed + base_speed
        v_close = -torch.sum(dv * u, dim=-1, keepdim=True) # (N, 1)
        max_v_close = self.physics.predator_sprint_speed + self.physics.base_speed
        v_close_norm = torch.clamp(v_close / max_v_close, min=-1.0, max=1.0)
        
        # 4. Looming (Time-to-collision proxy) (Normalized/Clamped)
        # v_close / d can explode if d is tiny. We clamp it realistically.
        loom = v_close / d.clamp(min=1e-5) # (N, 1)
        loom_norm = torch.clamp(loom, min=-5.0, max=5.0) / 5.0 # pseudo-normalized [-1, 1]
        
        # 5. Bearing (in_front) (Already Normalized -1 to 1 from dot product)
        # dot(self_vel_unit, u)
        self_vel_unit = vel / (torch.norm(vel, dim=-1, keepdim=True).clamp(min=1e-5))
        in_front = torch.sum(self_vel_unit * u, dim=-1, keepdim=True) # (N, 1)
        
        # Mask threats if far away (visual radius)
        threat_mask = d > (self.space_size / 2.0)
        v_close_norm.masked_fill_(threat_mask, 0.0)
        loom_norm.masked_fill_(threat_mask, 0.0)
        in_front.masked_fill_(threat_mask, 0.0)
        
        # === Bounds ===
        # 3D relative position from center [-1.0, 1.0]
        pos_relative = (pos - (self.space_size / 2.0)) / (self.space_size / 2.0)
        
        # Normalize self velocity [-1, 1]
        vel_norm = vel / self.physics.base_speed
        
        # Package for all agents identically so shapes NEVER change
        for i, agent in enumerate(self.possible_agents):
                
            obs = torch.cat([
                vel_norm[i],               # 3
                nearest_dist[i],           # 1
                local_density[i],          # 1
                local_alignment[i],        # 3
                com_direction[i],          # 3
                d_norm[i],                 # 1
                v_close_norm[i],           # 1
                loom_norm[i],              # 1
                in_front[i],               # 1
                pos_relative[i]            # 3
            ]).cpu().numpy()
            
            obs_dict[agent] = obs
            
        return obs_dict

    def _get_rewards(self):
        """
        Biological Phase 4 Reward:
        1. Survival Reward (+)
        2. Death Penalty (-) (Predator catch)
        3. Collision Penalty (-)
        4. Potential-Based Reward Shaping (PBRS) bounds & density
        """
        rewards = {}
        potentials = {}
        pos = self.physics.positions
        alive = self.physics.alive_mask
        
        # Pairwise distance
        dist_matrix = torch.cdist(pos, pos)
        dist_matrix.fill_diagonal_(float('inf'))
        
        # Collision: distance < 2.0 is bad
        collision_count = (dist_matrix < 2.0).sum(dim=1).float()
        
        # Density for PBRS
        dist_matrix_masked = dist_matrix.clone()
        dist_matrix_masked.masked_fill_(~alive.unsqueeze(0), float('inf'))
        in_radius_mask = (dist_matrix_masked < self.perception_radius) & alive.unsqueeze(0)
        local_density = in_radius_mask.sum(dim=1).float() / self.n_agents
        
        for i, agent in enumerate(self.possible_agents):
            if agent in self.dead_agents:
                 # Already processed the death in a previous frame
                 rewards[agent] = 0.0
                 potentials[agent] = 0.0
                 continue
            
            if not alive[i]:
                # Death by predator: First time seeing it dead
                rewards[agent] = -100.0
                potentials[agent] = 0.0
                self.dead_agents.add(agent)
                continue
                
            # Stay alive base reward
            rew = 0.1 
                 
            # Collision penalty
            if collision_count[i] > 0:
                 rew -= 2.0 * collision_count[i].item()
                 
            # PBRS Potentials
            rel_x = (pos[i, 0] - self.space_size / 2.0) / (self.space_size / 2.0)
            rel_y = (pos[i, 1] - self.space_size / 2.0) / (self.space_size / 2.0)
            rel_z = (pos[i, 2] - self.space_size / 2.0) / (self.space_size / 2.0)
            
            d_center_sq = rel_x**2 + rel_y**2 + rel_z**2
            phi_bounds = -self.pbrs_k * d_center_sq.item()
            phi_density = self.pbrs_c * local_density[i].item()
            
            potentials[agent] = phi_bounds + phi_density
            rewards[agent] = rew
             
        return rewards, potentials

    def render(self):
        pass

    def close(self):
        pass
