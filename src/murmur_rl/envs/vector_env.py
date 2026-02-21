import torch

from murmur_rl.envs.physics import BoidsPhysics


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
        space_size=100.0,
        perception_radius=10.0,
        device="cpu",
    ):
        self.n_agents = num_agents
        self.space_size = space_size
        self.perception_radius = perception_radius
        self.device = torch.device(device)

        self.physics = BoidsPhysics(
            num_boids=num_agents,
            space_size=space_size,
            device=self.device,
            perception_radius=perception_radius,
        )

        self.obs_dim = 16
        self.action_dim = 3
        self.num_moves = 0
        self.max_steps = 500

        # Track which agents have died (persistent across the episode)
        self._dead_mask = torch.zeros(
            num_agents, dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # --- Pairwise distances (shared with rewards) ---
        dist_matrix = torch.cdist(pos, pos)   # (N, N)
        dist_matrix.fill_diagonal_(float("inf"))
        dist_matrix.masked_fill_(~alive.unsqueeze(0), float("inf"))

        # === Group Context ===

        # 1. Nearest distance (N, 1)
        nearest_dist = dist_matrix.min(dim=1, keepdim=True).values
        nearest_dist = torch.where(
            nearest_dist == float("inf"),
            torch.tensor(self.perception_radius, device=self.device),
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
        pred_pos = self.physics.predator_position               # (1, 3)
        pred_vel = self.physics.predator_velocity               # (1, 3)

        dx = pred_pos - pos                                     # (N, 3)
        dv = pred_vel - vel                                     # (N, 3)

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
        far = d > (self.space_size / 2.0)
        v_close_norm = v_close_norm.masked_fill(far, 0.0)
        loom_norm = loom_norm.masked_fill(far, 0.0)
        in_front = in_front.masked_fill(far, 0.0)

        # === Boundary ===
        dist_lo = pos                                           # (N, 3)
        dist_hi = self.space_size - pos                         # (N, 3)
        closest_wall = torch.min(dist_lo, dist_hi).min(dim=1, keepdim=True).values
        closest_wall_norm = (closest_wall / (self.space_size / 2.0)).clamp(max=1.0)

        # === Velocity normalised ===
        vel_norm = vel / self.physics.base_speed

        # --- Concatenate (N, 16) ---
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
            closest_wall_norm,  # 1
        ], dim=1)

        return obs

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

        # Pairwise distances
        dist_matrix = torch.cdist(pos, pos)
        dist_matrix.fill_diagonal_(float("inf"))

        nearest_dist = dist_matrix.min(dim=1).values

        # Social: comfortable range (2, perception_radius]
        comfortable = (dist_matrix > 2.0) & (dist_matrix <= self.perception_radius)
        social_count = comfortable.sum(dim=1).float()

        # Collisions: < 1.0
        collision_count = (dist_matrix < 1.0).sum(dim=1).float()

        # Wall deaths
        margin = 1.0
        hit_wall = (pos < margin).any(dim=1) | (pos > (self.space_size - margin)).any(dim=1)

        # Predator deaths (physics already updated alive_mask)
        killed_by_predator = ~alive & ~self._dead_mask  # newly killed by predator

        # Wall deaths that are new
        killed_by_wall = hit_wall & ~self._dead_mask

        new_deaths = killed_by_predator | killed_by_wall

        # Kill wall-hit agents in physics too
        self.physics.alive_mask &= ~killed_by_wall

        # --- Compute rewards vectorised ---
        # Base survival reward
        rewards = torch.full((self.n_agents,), 0.1, device=self.device)

        # Social bonus: 0.05 * min(social_count, 5)
        rewards += 0.05 * social_count.clamp(max=5.0)

        # Collision penalty
        rewards -= 2.0 * collision_count

        # Death penalty overrides everything
        rewards = torch.where(new_deaths, torch.tensor(-100.0, device=self.device), rewards)

        # Already-dead agents get 0
        rewards = torch.where(self._dead_mask, torch.tensor(0.0, device=self.device), rewards)

        return rewards, new_deaths
