import torch

from murmur_rl.envs.physics import BoidsPhysics

# Detect best torch.compile mode at import time.
# On CUDA, the inductor backend requires Triton for kernel codegen.
# On CPU, the C++ codegen backend works without Triton.
_HAS_TRITON = False
try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    pass


class VectorMurmurationEnv:
    """
    Fully vectorized Murmuration environment. No PettingZoo, no dicts, no numpy.
    Everything stays as PyTorch tensors on-device for maximum throughput.

    Interface:
        reset()  -> obs (N, 16)
        step(actions: (N, 3))  -> obs (N, 16), rewards (N,), dones (N,)
    """

    def __init__(
        self,
        num_agents=50,
        num_predators=5,
        space_size=100.0,
        perception_radius=10.0,
        device="cpu",
    ):
        self.n_agents = num_agents
        self.num_predators = num_predators
        self.space_size = space_size
        self.perception_radius = perception_radius
        self.device = torch.device(device)

        self.physics = BoidsPhysics(
            num_boids=num_agents,
            num_predators=num_predators,
            space_size=space_size,
            device=self.device,
            perception_radius=perception_radius,
        )

        self.obs_dim = 18
        
        # Centralized Critic Global State Dimension:
        # ALL positions (N * 3) + ALL velocities (N * 3) + Predator Pos (P * 3) + Predator Vel (P * 3) + Alive Mask (N)
        self.global_obs_dim = (num_agents * 3) + (num_agents * 3) + (num_predators * 3) + (num_predators * 3) + num_agents
        
        self.action_dim = 3
        self.num_moves = 0
        self.max_steps = 500

        # Pre-compute constants as on-device tensors so compiled code
        # doesn't re-create them on every call
        self._perception_r = torch.tensor(perception_radius, device=self.device)
        self._half_space = torch.tensor(space_size / 2.0, device=self.device)
        self._death_penalty = torch.tensor(-100.0, device=self.device)
        self._zero = torch.tensor(0.0, device=self.device)
        self._inf = torch.tensor(float("inf"), device=self.device)

        # Diagonal mask for cdist — avoids in-place fill_diagonal_ which breaks torch.compile
        self._diag_mask = torch.eye(num_agents, dtype=torch.bool, device=self.device)

        # Track which agents have died (persistent across the episode)
        self._dead_mask = torch.zeros(
            num_agents, dtype=torch.bool, device=self.device
        )
        self.predator_danger_radius = 15.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self):
        """Wrap hot methods with torch.compile for kernel fusion.
        Call once after construction, before the training loop.
        """
        # CUDA inductor requires Triton; CPU can use C++ codegen (default).
        if self.device.type == "cuda" and not _HAS_TRITON:
            print("  env compile skipped (CUDA requires Triton, not installed)")
            return self

        mode = "reduce-overhead" if _HAS_TRITON else "default"
        try:
            self._get_observations = torch.compile(
                self._get_observations, mode=mode
            )
            self._get_rewards = torch.compile(
                self._get_rewards, mode=mode
            )
            print(f"  env compiled (mode={mode})")
        except Exception as e:
            print(f"  env compile skipped ({e})")
        return self

    def reset(self):
        self.num_moves = 0
        self.physics.reset()
        self._dead_mask.zero_()
        return self._get_observations()

    def step(self, actions: torch.Tensor):
        """
        Args:
            actions: (N, 3) tensor of normalized actions in [-1, 1]
        Returns:
            obs:   (N, 16) tensor
            rewards: (N,) tensor
            dones: (N,) bool tensor  (True = terminated or truncated)
        """
        # Scale actions and step physics
        scaled = actions * self.physics.max_force
        self.physics.step(actions=scaled)
        self.num_moves += 1

        obs = self._get_observations()
        rewards, new_deaths = self._get_rewards()

        # Update persistent death mask
        self._dead_mask |= new_deaths

        # Truncation
        truncated = self.num_moves >= self.max_steps
        if truncated:
            dones = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        else:
            dones = self._dead_mask.clone()

        return obs, rewards, dones

    # ------------------------------------------------------------------
    # Vectorized observations — zero Python loops
    # ------------------------------------------------------------------

    def _get_observations(self):
        pos = self.physics.positions          # (N, 3)
        vel = self.physics.velocities         # (N, 3)
        alive = self.physics.alive_mask       # (N,)

        # --- Pairwise distances — compile-friendly (no in-place ops) ---
        dist_matrix = torch.cdist(pos, pos)   # (N, N)
        # Replace diagonal + dead-agent entries with inf via torch.where
        mask_out = self._diag_mask | ~alive.unsqueeze(0)  # (N, N)
        dist_matrix = torch.where(mask_out, self._inf, dist_matrix)

        # === Group Context ===

        # 1. Nearest distance (N, 1)
        nearest_dist = dist_matrix.min(dim=1, keepdim=True).values
        nearest_dist = torch.where(
            nearest_dist == float("inf"),
            self._perception_r,
            nearest_dist,
        )
        nearest_dist = nearest_dist / self.perception_radius

        # 2. Local density (N, 1)
        in_radius = dist_matrix < self.perception_radius        # (N, N)
        in_radius_f = in_radius.float()                         # reuse below
        local_density = in_radius_f.sum(dim=1, keepdim=True) / self.n_agents

        # 3. Local alignment — vectorised via masked matmul
        neighbor_counts = in_radius_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        avg_vel = (in_radius_f @ vel) / neighbor_counts         # (N, 3)
        local_alignment = avg_vel / avg_vel.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        # 4. COM direction — vectorised via masked matmul
        avg_pos = (in_radius_f @ pos) / neighbor_counts         # (N, 3)
        dir_to_com = avg_pos - pos
        com_direction = dir_to_com / dir_to_com.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        # Zero out alignment/com for agents with no neighbours
        has_neighbors = in_radius.any(dim=1, keepdim=True)      # (N, 1)
        local_alignment = local_alignment * has_neighbors.float()
        com_direction = com_direction * has_neighbors.float()

        # === Perceptual Threat (Predator) ===
        pred_pos = self.physics.predator_position               # (num_predators, 3)
        pred_vel = self.physics.predator_velocity               # (num_predators, 3)

        # Find closest predator for each boid
        dist_to_preds = torch.cdist(pos, pred_pos)              # (N, num_predators)
        closest_pred_idx = torch.argmin(dist_to_preds, dim=1)   # (N,)
        
        closest_pred_pos = pred_pos[closest_pred_idx]           # (N, 3)
        closest_pred_vel = pred_vel[closest_pred_idx]           # (N, 3)

        dx = closest_pred_pos - pos                             # (N, 3)
        dv = closest_pred_vel - vel                             # (N, 3)

        d = dx.norm(dim=-1, keepdim=True)                       # (N, 1)
        d_norm = (d / (self.space_size / 2.0)).clamp(max=1.0)

        u = dx / d.clamp(min=1e-5)                              # (N, 3) unit dir

        # Closing speed
        max_v_close = self.physics.predator_speed + self.physics.base_speed
        v_close = -(dv * u).sum(dim=-1, keepdim=True)           # (N, 1)
        v_close_norm = (v_close / max_v_close).clamp(-1.0, 1.0)

        # Looming
        loom = v_close / d.clamp(min=1e-5)
        loom_norm = (loom / 5.0).clamp(-1.0, 1.0)

        # Bearing
        vel_unit = vel / vel.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        in_front = (vel_unit * u).sum(dim=-1, keepdim=True)     # (N, 1)

        # Mask far threats
        far = d > self._half_space
        v_close_norm = torch.where(far, self._zero, v_close_norm)
        loom_norm = torch.where(far, self._zero, loom_norm)
        in_front = torch.where(far, self._zero, in_front)

        # === Boundary ===
        # 3D relative position from center [-1.0, 1.0]
        pos_relative = (pos - self._half_space) / self._half_space

        # === Velocity normalised ===
        vel_norm = vel / self.physics.base_speed

        # --- Concatenate (N, 18) ---
        obs = torch.cat([
            vel_norm,           # 3
            nearest_dist,       # 1
            local_density,      # 1
            local_alignment,    # 3
            com_direction,      # 3
            d_norm,             # 1
            v_close_norm,       # 1
            loom_norm,          # 1
            in_front,           # 1
            pos_relative,       # 3
        ], dim=1)

        return obs
        
    def get_global_state(self):
        """
        Returns the global state tensor for the Centralized Critic.
        Shape: (N, global_obs_dim) -> Each agent gets a copy of the exact same global state.
        Contains flattened arrays of all boid positions, velocities, alive masks, and predator state.
        This provides perfect global omniscience to solve the Credit Assignment Problem.
        """
        pos = self.physics.positions.flatten() / self.space_size  # Normalized [0, 1]
        vel = self.physics.velocities.flatten() / self.physics.base_speed
        
        pred_pos = self.physics.predator_position.flatten() / self.space_size
        pred_vel = self.physics.predator_velocity.flatten() / self.physics.predator_speed
        
        alive = self.physics.alive_mask.float()
        
        # Concatenate everything into 1D global state vector
        global_state_1d = torch.cat([pos, vel, pred_pos, pred_vel, alive], dim=0)
        
        # Expand across all agents so each gets exactly the same global observation
        global_state = global_state_1d.unsqueeze(0).expand(self.n_agents, -1)
        
        return global_state

    # ------------------------------------------------------------------
    # Vectorized rewards — zero Python loops
    # ------------------------------------------------------------------

    def _get_rewards(self):
        """
        Returns:
            rewards:    (N,) float tensor
            new_deaths: (N,) bool tensor — agents that NEWLY died this step
        """
        pos = self.physics.positions
        alive = self.physics.alive_mask

        # Pairwise distances — compile-friendly
        dist_matrix = torch.cdist(pos, pos)
        dist_matrix = torch.where(self._diag_mask, self._inf, dist_matrix)

        nearest_dist = dist_matrix.min(dim=1).values

        # Social: comfortable range (2, perception_radius]
        comfortable = (dist_matrix > 2.0) & (dist_matrix <= self.perception_radius)
        social_count = comfortable.sum(dim=1).float()

        # Collisions: < 1.0
        collision_count = (dist_matrix < 1.0).sum(dim=1).float()

        # Predator deaths (physics already updated alive_mask)
        killed_by_predator = ~alive & ~self._dead_mask  # newly killed by predator
        new_deaths = killed_by_predator

        # Continuous boundary penalty
        margin = self.space_size * 0.1
        
        # Original XY boundaries
        dist_lo_xy = pos[:, :2]
        dist_hi_xy = self.space_size - pos[:, :2]
        closest_wall_xy = torch.min(dist_lo_xy, dist_hi_xy).min(dim=1).values
        penetration_xy = torch.clamp(margin - closest_wall_xy, min=0.0)
        boundary_penalty_xy = -10.0 * (penetration_xy / margin)
        
        # New Z-bound penalty (Floor=0, Ceiling=0.85)
        z_pos = pos[:, 2]
        ceiling = self.space_size * 0.85
        dist_lo_z = z_pos
        dist_hi_z = ceiling - z_pos
        
        # Penalize approaching the floor or the lowered ceiling
        closest_wall_z = torch.min(dist_lo_z, dist_hi_z)
        penetration_z = torch.clamp(margin - closest_wall_z, min=0.0)
        
        # Make the ceiling penalty much steeper to discourage hiding near predators
        is_ceiling = dist_hi_z < dist_lo_z
        penalty_z_mult = torch.where(is_ceiling, 20.0, 10.0)
        boundary_penalty_z = -penalty_z_mult * (penetration_z / margin)
        
        boundary_penalty = boundary_penalty_xy + boundary_penalty_z

        # --- Compute rewards vectorised ---
        # Base survival reward
        rewards = torch.full((self.n_agents,), 0.1, device=self.device)

        # Collision penalty
        rewards -= 2.0 * collision_count

        # Apply continuous boundary penalty
        rewards += boundary_penalty

        # Death penalty overrides everything
        rewards = torch.where(new_deaths, self._death_penalty, rewards)

        # Already-dead agents get 0
        rewards = torch.where(self._dead_mask, self._zero, rewards)

        return rewards, new_deaths
