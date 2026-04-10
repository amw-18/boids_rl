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
        reset()  -> obs (N, 18)
        step(actions: (N, 3))  -> obs (N, 18), rewards (N,), dones (N,)
    """

    def __init__(
        self,
        num_agents=50,
        num_predators=5,
        space_size=100.0,
        perception_radius=10.0,
        base_speed=5.0,
        max_turn_angle=0.5,
        max_force=2.0,
        device="cpu",
        gamma=0.99,
        pbrs_k=1.0,
        pbrs_c=1.0,
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
            base_speed=base_speed,
            max_turn_angle=max_turn_angle,
            max_force=max_force,
        )

        self.obs_dim = 18
        self.pred_obs_dim = 10 + min(5, self.n_agents) * 7
        
        K = min(10, self.n_agents)
        self.global_obs_dim = self.obs_dim + (K * 7) + (num_predators * 6)
        self.pred_global_obs_dim = self.pred_obs_dim + (K * 7) + (num_predators * 6)
        
        self.action_dim = 3
        self.num_moves = 0
        self.env_step_counter = 0
        self.max_steps = 500

        # Pre-compute constants as on-device tensors so compiled code
        # doesn't re-create them on every call
        self._perception_r = torch.tensor(perception_radius, device=self.device)
        self._half_space = torch.tensor(space_size / 2.0, device=self.device)
        self._death_penalty = torch.tensor(-100.0, device=self.device)
        self._zero = torch.tensor(0.0, device=self.device)
        self._inf = torch.tensor(float("inf"), device=self.device)

        self._gamma = torch.tensor(gamma, device=self.device)
        self._pbrs_k = torch.tensor(pbrs_k, device=self.device)
        self._pbrs_c = torch.tensor(pbrs_c, device=self.device)
        self.last_potential = torch.zeros(num_agents, device=self.device)
        self.last_pred_potential = torch.zeros(num_predators, device=self.device)

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
        _, _, _, new_potential, new_pred_potential = self._get_rewards()
        self.last_potential = new_potential
        self.last_pred_potential = new_pred_potential
        return self._get_observations(), self._get_predator_observations()

    def step(self, boid_actions: torch.Tensor, predator_actions: torch.Tensor = None):
        """
        Args:
            boid_actions: (N, 3) tensor
            predator_actions: (P, 3) tensor
        Returns:
            obs_boids:   (N, 18) tensor
            obs_preds:   (P, pred_obs_dim) tensor
            rewards_boids: (N,) tensor
            rewards_preds: (P,) tensor
            dones: (N,) bool tensor  (True = terminated or truncated)
        """
        # Step physics using Thrust, Roll, Pitch actions [-1, 1]
        self.physics.step(boid_actions=boid_actions, predator_actions=predator_actions)
        self.num_moves += 1
        
        # Curriculum: Anneal predator catch radius from 2.0 to 0.5 over 5M steps
        self.env_step_counter += 1
        decay_steps = 5000000.0
        progress = min(1.0, self.env_step_counter / decay_steps)
        self.physics.predator_catch_radius = 2.0 - progress * 1.5

        obs_boids = self._get_observations()
        obs_preds = self._get_predator_observations()
        rewards_boids, rewards_preds, new_deaths, new_potential, new_pred_potential = self._get_rewards()

        # PBRS shaping for Boids
        shaping = (self._gamma * new_potential) - self.last_potential
        rewards_boids += shaping
        self.last_potential = new_potential.clone()

        # PBRS shaping for Predators
        pred_shaping = (self._gamma * new_pred_potential) - self.last_pred_potential
        rewards_preds += pred_shaping
        self.last_pred_potential = new_pred_potential.clone()

        # Update persistent death mask
        self._dead_mask |= new_deaths

        # Truncation
        truncated = self.num_moves >= self.max_steps
        if truncated:
            dones = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        else:
            dones = self._dead_mask.clone()

        return obs_boids, obs_preds, rewards_boids, rewards_preds, dones

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
        is_active = has_neighbors & alive.unsqueeze(1)
        local_alignment = local_alignment * is_active.float()
        com_direction = com_direction * is_active.float()

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
        max_v_close = self.physics.predator_sprint_speed + self.physics.base_speed
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

    def _get_predator_observations(self):
        """
        Build local observation vectors for the RL Predators.
        Features: (Own Kinematics) + (Mean Field CoM) + (k Closest Starlings w/ Visual Obfuscation),
        where k=min(5, num_agents).
        """
        pred_pos = self.physics.predator_position # (P, 3)
        pred_vel = self.physics.predator_velocity # (P, 3)
        boid_pos = self.physics.positions # (N, 3)
        alive = self.physics.alive_mask # (N,)

        # Own Kinematics
        pos_relative = (pred_pos - self._half_space) / self._half_space # (P, 3)
        vel_norm = pred_vel / self.physics.predator_sprint_speed # (P, 3)
        stamina_norm = (self.physics.predator_stamina / self.physics.predator_max_stamina).unsqueeze(1) # (P, 1)

        # Center of Mass of Alive Swarm
        alive_f = alive.float().unsqueeze(1)
        num_alive = alive_f.sum().clamp(min=1.0)
        com = (boid_pos * alive_f).sum(dim=0) / num_alive # (3,)
        com_relative = (com - pred_pos) / self._half_space # (P, 3)

        # Visual Obfuscation: Find 5 closest alive targets
        dist_matrix = torch.cdist(pred_pos, boid_pos) # (P, N)
        # Mask out dead boids
        dist_matrix = torch.where(alive.unsqueeze(0), dist_matrix, self._inf)

        k = min(5, self.n_agents)
        closest_dists, closest_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False) # (P, k)

        target_obs = []
        for i in range(k):
            target_ids = closest_idx[:, i] # (P,)
            dists = closest_dists[:, i:i+1] # (P, 1)

            # If the closest target is dead (happens if < k alive), zero out its obs
            is_valid = (dists < self._inf).float()

            target_positions = boid_pos[target_ids] # (P, 3)
            target_velocities = self.physics.velocities[target_ids] # (P, 3)

            # --- CALCULATE VISUAL OBFUSCATION NOISE ---
            # We need the local density of each of these k targets
            # To do this safely/fast, we grab their row from the boid-boid distance matrix
            b_b_dist = torch.cdist(target_positions, boid_pos) # (P, N)
            b_b_dist = torch.where(alive.unsqueeze(0), b_b_dist, self._inf)
            target_density = (b_b_dist < self.perception_radius).float().sum(dim=1, keepdim=True) / self.n_agents # (P, 1)

            # Gaussian Noise explicitly scales with density.
            # Tuning param: a density of 1.0 (everyone) causes max_noise variance
            max_noise_variance = 5.0 # units of spatial distortion
            sigma = target_density * max_noise_variance
            noise = torch.randn_like(target_positions) * sigma

            obfuscated_target_pos = target_positions + noise
            rel_pos = (obfuscated_target_pos - pred_pos) / self._half_space # (P, 3)
            rel_vel = (target_velocities - pred_vel) / (self.physics.predator_sprint_speed + self.physics.base_speed) # (P, 3)

            target_obs.append(rel_pos * is_valid)
            target_obs.append(rel_vel * is_valid)
            target_obs.append((dists / (self.space_size * 1.5)) * is_valid) # (P, 1) Normalized distance

        # Concat all elements
        target_obs_tensor = torch.cat(target_obs, dim=1) if len(target_obs) > 0 else torch.zeros((self.num_predators, 0), device=self.device)

        # Total Predator Obs: (P, 10 + k * 7), where k=min(5, num_agents)
        obs = torch.cat([pos_relative, vel_norm, stamina_norm, com_relative, target_obs_tensor], dim=1)
        return obs

    def _build_global_state(self, local_obs, focal_pos, *, exclude_self):
        """Build the centralized critic state from an explicit focal-agent population."""
        batch_size = local_obs.shape[0]
        pos = self.physics.positions # (N, 3)
        vel = self.physics.velocities # (N, 3)
        alive = self.physics.alive_mask # (N,)

        K = min(10, self.n_agents)
        dist_matrix = torch.cdist(focal_pos, pos) # (batch_size, N)
        dist_matrix = torch.where(alive.unsqueeze(0), dist_matrix, self._inf)

        if exclude_self:
            # Mask out self
            dist_matrix = torch.where(self._diag_mask, self._inf, dist_matrix)
            
        _, closest_idx = torch.topk(dist_matrix, k=K, dim=1, largest=False) # (batch_size, K)
        
        # Gather K features
        k_pos = pos[closest_idx] # (batch_size, K, 3)
        k_vel = vel[closest_idx] # (batch_size, K, 3)
        
        # We must mask out the focal agent's alive flag if it gets pulled into the 
        # padding slots due to a small swarm size.
        k_alive = alive[closest_idx].float() # (batch_size, K)
        if exclude_self:
            focal_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
            k_alive = torch.where(closest_idx == focal_idx, 0.0, k_alive)
        k_alive = k_alive.unsqueeze(-1) # (batch_size, K, 1)
        
        rel_pos = ((k_pos - focal_pos.unsqueeze(1)) / self._half_space) * k_alive
        rel_vel = (k_vel / self.physics.base_speed) * k_alive
        
        k_features = torch.cat([rel_pos, rel_vel, k_alive], dim=-1) # (batch_size, K, 7)
        k_features_flat = k_features.view(batch_size, -1) # (batch_size, K * 7)
        
        # Predator features
        pred_pos = self.physics.predator_position # (P, 3)
        pred_vel = self.physics.predator_velocity # (P, 3)
        
        rel_pred_pos = (pred_pos.unsqueeze(0) - focal_pos.unsqueeze(1)) / self._half_space # (batch_size, P, 3)
        rel_pred_vel = pred_vel.unsqueeze(0).expand(batch_size, -1, -1) / self.physics.predator_sprint_speed # (batch_size, P, 3)
        
        pred_features = torch.cat([rel_pred_pos, rel_pred_vel], dim=-1) # (batch_size, P, 6)
        pred_features_flat = pred_features.view(batch_size, -1) # (batch_size, P * 6)
        
        global_state = torch.cat([local_obs, k_features_flat, pred_features_flat], dim=1)
        return global_state

    def get_boid_global_state(self, local_obs):
        if local_obs.shape[0] != self.n_agents:
            raise ValueError("boid global state expects one local observation per boid.")
        return self._build_global_state(local_obs, self.physics.positions, exclude_self=True)

    def get_predator_global_state(self, local_obs):
        if local_obs.shape[0] != self.num_predators:
            raise ValueError("predator global state expects one local observation per predator.")
        return self._build_global_state(local_obs, self.physics.predator_position, exclude_self=False)

    def get_global_state(self, local_obs, agent_type=None):
        """
        Backwards-compatible wrapper for centralized critic state construction.
        """
        if agent_type == "boid":
            return self.get_boid_global_state(local_obs)
        if agent_type == "predator":
            return self.get_predator_global_state(local_obs)

        batch_size = local_obs.shape[0]
        if batch_size == self.n_agents and self.n_agents != self.num_predators:
            return self.get_boid_global_state(local_obs)
        if batch_size == self.num_predators and self.n_agents != self.num_predators:
            return self.get_predator_global_state(local_obs)

        raise ValueError(
            "agent_type must be provided when local_obs batch size is ambiguous."
        )

    # ------------------------------------------------------------------
    # Vectorized rewards — zero Python loops
    # ------------------------------------------------------------------

    def _get_rewards(self):
        """
        Returns:
            rewards:       (N,) float tensor
            new_deaths:    (N,) bool tensor — agents that NEWLY died this step
            new_potential: (N,) float tensor — PBRS potential
        """
        pos = self.physics.positions
        alive = self.physics.alive_mask

        # Pairwise distances — compile-friendly
        dist_matrix = torch.cdist(pos, pos)
        dist_matrix = torch.where(self._diag_mask, self._inf, dist_matrix)
        live_pair_mask = alive.unsqueeze(0) & alive.unsqueeze(1)
        live_dist_matrix = torch.where(live_pair_mask, dist_matrix, self._inf)

        # Collisions: < 2.0
        collision_count = (live_dist_matrix < 2.0).sum(dim=1).float()

        # Predator deaths (physics already updated alive_mask)
        new_deaths = self.physics.last_capture_mask

        # Potential-Based Reward Shaping (PBRS)
        
        # 1. Boundary Potential (phi_bounds)
        pos_relative = (pos - self._half_space) / self._half_space
        d_center_sq = (pos_relative**2).sum(dim=-1)
        phi_bounds = -self._pbrs_k * d_center_sq
        
        # 2. Density Potential (phi_density) — must mask dead agents like PZ env
        in_radius = live_dist_matrix < self.perception_radius
        local_density = in_radius.float().sum(dim=1) / self.n_agents
        phi_density = self._pbrs_c * local_density
        
        new_potential = phi_bounds + phi_density

        # --- Compute rewards vectorised ---
        # Base survival reward
        rewards = torch.full((self.n_agents,), 0.1, device=self.device)

        # Collision penalty
        rewards -= 2.0 * collision_count

        # Death penalty overrides everything
        rewards = torch.where(new_deaths, self._death_penalty, rewards)

        # Already-dead agents get 0
        rewards = torch.where(self._dead_mask, self._zero, rewards)

        # Terminal states must have 0 potential
        new_potential = torch.where(self._dead_mask | new_deaths, self._zero, new_potential)

        # Predator Rewards with PBRS boundary potential
        pred_pos = self.physics.predator_position  # (P, 3)
        pred_pos_relative = (pred_pos - self._half_space) / self._half_space
        pred_d_center_sq = (pred_pos_relative**2).sum(dim=-1)  # (P,)
        pred_phi_bounds = -self._pbrs_k * pred_d_center_sq  # (P,)

        rewards_preds = torch.zeros(self.num_predators, device=self.device)

        # Catch reward: +10.0 per boid caught this step
        catches_per_pred = self.physics.last_capture_counts
        rewards_preds += 10.0 * catches_per_pred

        # Hunger penalty: constant penalty per step to encourage quick catches
        is_cooldown = self.physics.predator_cooldown > 0
        made_catch = catches_per_pred > 0
        hunger_penalty = -0.05
        
        rewards_preds += torch.where(
            ~is_cooldown & ~made_catch,
            torch.tensor(hunger_penalty, device=self.device),
            self._zero
        )

        return rewards, rewards_preds, new_deaths, new_potential, pred_phi_bounds
