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
        min_speed=2.5,
        dt=0.1,
        device="cpu",
        gamma=0.99,
        boundary_mode="legacy_pbrs",
        pbrs_k=1.0,
        pbrs_c=1.0,
        wall_soft_margin=0.0,
        wall_penalty=0.0,
        predator_wall_penalty=0.0,
        wall_bounce_damping=1.0,
        max_steps=500,
        curriculum_enabled=True,
        predator_catch_radius_start=2.0,
        predator_catch_radius_end=0.5,
        predator_catch_radius_decay_steps=5000000,
        predator_visual_noise_variance=5.0,
        predator_sprint_multiplier=1.5,
        predator_turn_multiplier=1.5,
        predator_cooldown_duration=50,
        predator_max_stamina=100.0,
        predator_sprint_drain=1.0,
        predator_recovery_rate=0.5,
        survival_reward=0.1,
        collision_penalty=2.0,
        death_penalty=-100.0,
        predator_catch_reward=10.0,
        predator_team_catch_reward=0.0,
        predator_hunger_penalty=-0.05,
    ):
        if boundary_mode not in {"legacy_pbrs", "hard_walls"}:
            raise ValueError(f"Unsupported boundary_mode: {boundary_mode}")

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
            min_speed=min_speed,
            dt=dt,
            predator_sprint_multiplier=predator_sprint_multiplier,
            predator_turn_multiplier=predator_turn_multiplier,
            predator_catch_radius=predator_catch_radius_start,
            predator_max_stamina=predator_max_stamina,
            predator_sprint_drain=predator_sprint_drain,
            predator_recovery_rate=predator_recovery_rate,
            predator_cooldown_duration=predator_cooldown_duration,
            enforce_hard_walls=(boundary_mode == "hard_walls"),
            wall_bounce_damping=wall_bounce_damping,
        )

        self.obs_dim = 18
        self.pred_obs_dim = 10 + min(5, self.n_agents) * 7
        
        K = min(10, self.n_agents)
        self.global_obs_dim = self.obs_dim + (K * 7) + (num_predators * 6)
        self.pred_global_obs_dim = self.pred_obs_dim + (K * 7) + (num_predators * 6)
        
        self.action_dim = 3
        self.num_moves = 0
        self.env_step_counter = 0
        self.max_steps = max_steps
        self.boundary_mode = boundary_mode
        self.curriculum_enabled = curriculum_enabled
        self.predator_catch_radius_start = predator_catch_radius_start
        self.predator_catch_radius_end = predator_catch_radius_end
        self.predator_catch_radius_decay_steps = predator_catch_radius_decay_steps
        self.predator_visual_noise_variance = predator_visual_noise_variance
        self.survival_reward = survival_reward
        self.collision_penalty = collision_penalty
        self.wall_soft_margin = wall_soft_margin
        self.wall_penalty = wall_penalty
        self.predator_wall_penalty = predator_wall_penalty
        self.predator_catch_reward = predator_catch_reward
        self.predator_team_catch_reward = predator_team_catch_reward
        self.predator_hunger_penalty = predator_hunger_penalty

        # Pre-compute constants as on-device tensors so compiled code
        # doesn't re-create them on every call
        self._perception_r = torch.tensor(perception_radius, device=self.device)
        self._half_space = torch.tensor(space_size / 2.0, device=self.device)
        self._death_penalty = torch.tensor(death_penalty, device=self.device)
        self._survival_reward = torch.tensor(survival_reward, device=self.device)
        self._collision_penalty = torch.tensor(collision_penalty, device=self.device)
        self._wall_soft_margin = torch.tensor(max(wall_soft_margin, 0.0), device=self.device)
        self._wall_penalty = torch.tensor(wall_penalty, device=self.device)
        self._predator_wall_penalty = torch.tensor(predator_wall_penalty, device=self.device)
        self._predator_catch_reward = torch.tensor(predator_catch_reward, device=self.device)
        self._predator_team_catch_reward = torch.tensor(predator_team_catch_reward, device=self.device)
        self._predator_hunger_penalty = torch.tensor(predator_hunger_penalty, device=self.device)
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
        self._step_cache = None
        self._step_cache_signature = None
        self.predator_danger_radius = 15.0
        self._apply_curriculum()

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
        self._apply_curriculum()
        self._invalidate_step_cache()
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
        
        self.env_step_counter += 1
        self._apply_curriculum()
        self._invalidate_step_cache()

        obs_boids = self._get_observations()
        obs_preds = self._get_predator_observations()
        rewards_boids, rewards_preds, new_deaths, new_potential, new_pred_potential = self._get_rewards()

        episode_over = torch.all(self._dead_mask | new_deaths)
        truncated = self.num_moves >= self.max_steps
        if episode_over or truncated:
            effective_new_potential = torch.zeros_like(new_potential)
            effective_new_pred_potential = torch.zeros_like(new_pred_potential)
        else:
            effective_new_potential = new_potential
            effective_new_pred_potential = new_pred_potential

        # PBRS shaping for Boids
        shaping = (self._gamma * effective_new_potential) - self.last_potential
        rewards_boids += shaping
        self.last_potential = effective_new_potential.clone()

        # PBRS shaping for Predators
        pred_shaping = (self._gamma * effective_new_pred_potential) - self.last_pred_potential
        rewards_preds += pred_shaping
        self.last_pred_potential = effective_new_pred_potential.clone()

        # Update persistent death mask
        self._dead_mask |= new_deaths

        # Truncation
        if truncated:
            dones = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        else:
            dones = self._dead_mask.clone()

        return obs_boids, obs_preds, rewards_boids, rewards_preds, dones

    def _invalidate_step_cache(self):
        self._step_cache = None
        self._step_cache_signature = None

    def _tensor_signature(self, tensor: torch.Tensor):
        return (id(tensor), getattr(tensor, "_version", 0))

    def _current_cache_signature(self):
        return (
            self._tensor_signature(self.physics.positions),
            self._tensor_signature(self.physics.velocities),
            self._tensor_signature(self.physics.predator_position),
            self._tensor_signature(self.physics.predator_velocity),
            self._tensor_signature(self.physics.alive_mask),
            self._tensor_signature(self.physics.predator_stamina),
        )

    def _build_step_cache(self):
        pos = self.physics.positions
        alive = self.physics.alive_mask
        alive_f = alive.to(dtype=pos.dtype)

        boid_boid_dist = torch.cdist(pos, pos)
        live_column_mask = alive.unsqueeze(0)
        boid_neighbor_dist = torch.where(
            self._diag_mask | ~live_column_mask,
            self._inf,
            boid_boid_dist,
        )

        live_pair_mask = alive.unsqueeze(0) & alive.unsqueeze(1)
        live_boid_dist = torch.where(
            self._diag_mask | ~live_pair_mask,
            self._inf,
            boid_boid_dist,
        )

        in_radius = boid_neighbor_dist < self.perception_radius
        in_radius_f = in_radius.to(dtype=pos.dtype)
        neighbor_counts = in_radius_f.sum(dim=1, keepdim=True)
        neighbor_counts_clamped = neighbor_counts.clamp(min=1.0)
        local_density = neighbor_counts / self.n_agents

        nearest_neighbor_dist = boid_neighbor_dist.min(dim=1, keepdim=True).values
        nearest_neighbor_dist = torch.where(
            nearest_neighbor_dist == float("inf"),
            self._perception_r,
            nearest_neighbor_dist,
        )

        alive_weights = alive_f.unsqueeze(1)
        alive_count = alive_weights.sum().clamp(min=1.0)
        alive_center_of_mass = (pos * alive_weights).sum(dim=0) / alive_count
        target_density_with_self = local_density.squeeze(1) + (alive_f / self.n_agents)

        K = min(10, self.n_agents)
        _, boid_global_idx = torch.topk(
            boid_neighbor_dist,
            k=K,
            dim=1,
            largest=False,
        )

        predator_obs_k = min(5, self.n_agents)
        if self.num_predators > 0:
            boid_pred_dist = torch.cdist(pos, self.physics.predator_position)
            closest_pred_idx = torch.argmin(boid_pred_dist, dim=1)
            closest_pred_dist = boid_pred_dist.gather(1, closest_pred_idx.unsqueeze(1))

            pred_boid_live_dist = torch.where(
                live_column_mask,
                boid_pred_dist.transpose(0, 1),
                self._inf,
            )
            pred_target_dists, pred_target_idx = torch.topk(
                pred_boid_live_dist,
                k=predator_obs_k,
                dim=1,
                largest=False,
            )
            _, pred_global_idx = torch.topk(
                pred_boid_live_dist,
                k=K,
                dim=1,
                largest=False,
            )
        else:
            boid_pred_dist = pos.new_empty((self.n_agents, 0))
            closest_pred_idx = torch.empty((self.n_agents,), dtype=torch.long, device=self.device)
            closest_pred_dist = pos.new_empty((self.n_agents, 1))
            pred_boid_live_dist = pos.new_empty((0, self.n_agents))
            pred_target_dists = pos.new_empty((0, predator_obs_k))
            pred_target_idx = torch.empty((0, predator_obs_k), dtype=torch.long, device=self.device)
            pred_global_idx = torch.empty((0, K), dtype=torch.long, device=self.device)

        return {
            "boid_neighbor_dist": boid_neighbor_dist,
            "live_boid_dist": live_boid_dist,
            "in_radius_f": in_radius_f,
            "neighbor_counts_clamped": neighbor_counts_clamped,
            "has_neighbors": in_radius.any(dim=1, keepdim=True),
            "nearest_neighbor_dist": nearest_neighbor_dist,
            "local_density": local_density,
            "alive_center_of_mass": alive_center_of_mass,
            "target_density_with_self": target_density_with_self,
            "closest_pred_idx": closest_pred_idx,
            "closest_pred_dist": closest_pred_dist,
            "pred_target_dists": pred_target_dists,
            "pred_target_idx": pred_target_idx,
            "boid_global_idx": boid_global_idx,
            "pred_global_idx": pred_global_idx,
        }

    def _ensure_step_cache(self):
        signature = self._current_cache_signature()
        if self._step_cache is not None and self._step_cache_signature == signature:
            return self._step_cache

        self._step_cache = self._build_step_cache()
        self._step_cache_signature = signature
        return self._step_cache

    def _wall_pressure(self, positions: torch.Tensor) -> torch.Tensor:
        if self._wall_soft_margin.item() <= 0.0:
            return torch.zeros(positions.shape[0], device=self.device, dtype=positions.dtype)

        distance_to_low = positions
        distance_to_high = self.space_size - positions
        nearest_wall = torch.minimum(distance_to_low, distance_to_high).min(dim=1).values
        penetration = ((self._wall_soft_margin - nearest_wall) / self._wall_soft_margin).clamp(min=0.0, max=1.0)
        return penetration.square()

    # ------------------------------------------------------------------
    # Vectorized observations — zero Python loops
    # ------------------------------------------------------------------

    def _get_observations(self):
        pos = self.physics.positions          # (N, 3)
        vel = self.physics.velocities         # (N, 3)
        alive = self.physics.alive_mask       # (N,)

        # --- Pairwise distances — compile-friendly (no in-place ops) ---
        cache = self._ensure_step_cache()
        nearest_dist = cache["nearest_neighbor_dist"] / self.perception_radius
        local_density = cache["local_density"]
        in_radius_f = cache["in_radius_f"]

        # 3. Local alignment — vectorised via masked matmul
        neighbor_counts = cache["neighbor_counts_clamped"]
        avg_vel = (in_radius_f @ vel) / neighbor_counts         # (N, 3)
        local_alignment = avg_vel / avg_vel.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        # 4. COM direction — vectorised via masked matmul
        avg_pos = (in_radius_f @ pos) / neighbor_counts         # (N, 3)
        dir_to_com = avg_pos - pos
        com_direction = dir_to_com / dir_to_com.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        # Zero out alignment/com for agents with no neighbours
        has_neighbors = cache["has_neighbors"]                  # (N, 1)
        is_active = has_neighbors & alive.unsqueeze(1)
        local_alignment = local_alignment * is_active.float()
        com_direction = com_direction * is_active.float()

        # === Perceptual Threat (Predator) ===
        if self.num_predators > 0:
            pred_pos = self.physics.predator_position               # (num_predators, 3)
            pred_vel = self.physics.predator_velocity               # (num_predators, 3)

            closest_pred_idx = cache["closest_pred_idx"]            # (N,)
            closest_pred_pos = pred_pos[closest_pred_idx]           # (N, 3)
            closest_pred_vel = pred_vel[closest_pred_idx]           # (N, 3)

            dx = closest_pred_pos - pos                             # (N, 3)
            dv = closest_pred_vel - vel                             # (N, 3)

            d = cache["closest_pred_dist"]                          # (N, 1)
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
        else:
            d_norm = torch.zeros((self.n_agents, 1), device=self.device)
            v_close_norm = torch.zeros((self.n_agents, 1), device=self.device)
            loom_norm = torch.zeros((self.n_agents, 1), device=self.device)
            in_front = torch.zeros((self.n_agents, 1), device=self.device)

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
        boid_vel = self.physics.velocities # (N, 3)
        cache = self._ensure_step_cache()

        # Own Kinematics
        pos_relative = (pred_pos - self._half_space) / self._half_space # (P, 3)
        vel_norm = pred_vel / self.physics.predator_sprint_speed # (P, 3)
        stamina_norm = (self.physics.predator_stamina / self.physics.predator_max_stamina).unsqueeze(1) # (P, 1)

        # Center of Mass of Alive Swarm
        com = cache["alive_center_of_mass"] # (3,)
        com_relative = (com - pred_pos) / self._half_space # (P, 3)

        # Visual Obfuscation: Find 5 closest alive targets
        k = min(5, self.n_agents)
        closest_dists = cache["pred_target_dists"] # (P, k)
        closest_idx = cache["pred_target_idx"] # (P, k)
        is_valid = (closest_dists < self._inf).unsqueeze(-1).float()

        target_positions = boid_pos[closest_idx] # (P, k, 3)
        target_velocities = boid_vel[closest_idx] # (P, k, 3)
        target_density = cache["target_density_with_self"][closest_idx].unsqueeze(-1) # (P, k, 1)

        # Preserve the legacy RNG consumption order for fixed-seed equivalence.
        noise = torch.stack(
            [
                torch.randn(
                    (self.num_predators, 3),
                    device=self.device,
                    dtype=target_positions.dtype,
                )
                for _ in range(k)
            ],
            dim=1,
        )
        sigma = target_density * self.predator_visual_noise_variance
        obfuscated_target_pos = target_positions + (noise * sigma)

        rel_pos = (obfuscated_target_pos - pred_pos.unsqueeze(1)) / self._half_space # (P, k, 3)
        rel_vel = (target_velocities - pred_vel.unsqueeze(1)) / (self.physics.predator_sprint_speed + self.physics.base_speed) # (P, k, 3)
        dist_feature = torch.where(
            is_valid.bool(),
            closest_dists.unsqueeze(-1) / (self.space_size * 1.5),
            torch.zeros_like(closest_dists.unsqueeze(-1)),
        )

        target_obs_tensor = torch.cat(
            [
                rel_pos * is_valid,
                rel_vel * is_valid,
                dist_feature * is_valid,
            ],
            dim=-1,
        ).reshape(self.num_predators, k * 7)

        # Total Predator Obs: (P, 10 + k * 7), where k=min(5, num_agents)
        obs = torch.cat([pos_relative, vel_norm, stamina_norm, com_relative, target_obs_tensor], dim=1)
        return obs

    def _build_global_state(self, local_obs, focal_pos, *, exclude_self):
        """Build the centralized critic state from an explicit focal-agent population."""
        batch_size = local_obs.shape[0]
        pos = self.physics.positions # (N, 3)
        vel = self.physics.velocities # (N, 3)
        alive = self.physics.alive_mask # (N,)
        cache = self._ensure_step_cache()
        closest_idx = cache["boid_global_idx"] if exclude_self else cache["pred_global_idx"]
        
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
        cache = self._ensure_step_cache()
        live_dist_matrix = cache["live_boid_dist"]

        # Collisions: < 2.0
        collision_count = (live_dist_matrix < 2.0).sum(dim=1).float()

        # Predator deaths (physics already updated alive_mask)
        new_deaths = self.physics.last_capture_mask

        # Potential-Based Reward Shaping (PBRS)
        
        # 1. Boundary Potential (phi_bounds)
        if self.boundary_mode == "legacy_pbrs":
            pos_relative = (pos - self._half_space) / self._half_space
            d_center_sq = (pos_relative**2).sum(dim=-1)
            phi_bounds = -self._pbrs_k * d_center_sq
        else:
            phi_bounds = torch.zeros(self.n_agents, device=self.device)
        
        # 2. Density Potential (phi_density) — must mask dead agents like PZ env
        local_density = (live_dist_matrix < self.perception_radius).float().sum(dim=1) / self.n_agents
        phi_density = self._pbrs_c * local_density
        
        new_potential = phi_bounds + phi_density

        # --- Compute rewards vectorised ---
        # Base survival reward
        rewards = torch.full((self.n_agents,), self._survival_reward.item(), device=self.device)

        # Collision penalty
        rewards -= self._collision_penalty * collision_count
        if self.boundary_mode == "hard_walls" and self._wall_penalty.item() > 0.0:
            rewards -= self._wall_penalty * self._wall_pressure(pos)

        # Death penalty overrides everything
        rewards = torch.where(new_deaths, self._death_penalty, rewards)

        # Already-dead agents get 0
        rewards = torch.where(self._dead_mask, self._zero, rewards)

        # Terminal states must have 0 potential
        new_potential = torch.where(self._dead_mask | new_deaths, self._zero, new_potential)

        # Predator Rewards with PBRS boundary potential
        pred_pos = self.physics.predator_position  # (P, 3)
        if self.boundary_mode == "legacy_pbrs":
            pred_pos_relative = (pred_pos - self._half_space) / self._half_space
            pred_d_center_sq = (pred_pos_relative**2).sum(dim=-1)  # (P,)
            pred_phi_bounds = -self._pbrs_k * pred_d_center_sq  # (P,)
        else:
            pred_phi_bounds = torch.zeros(self.num_predators, device=self.device)

        rewards_preds = torch.zeros(self.num_predators, device=self.device)

        # Catch reward: +10.0 per boid caught this step
        catches_per_pred = self.physics.last_capture_counts
        rewards_preds += self._predator_catch_reward * catches_per_pred
        if self._predator_team_catch_reward.item() != 0.0:
            rewards_preds += self._predator_team_catch_reward * catches_per_pred.sum()

        # Hunger penalty: constant penalty per step to encourage quick catches
        is_cooldown = self.physics.predator_cooldown > 0
        made_catch = catches_per_pred > 0
        rewards_preds += torch.where(
            ~is_cooldown & ~made_catch,
            self._predator_hunger_penalty,
            self._zero
        )
        if self.boundary_mode == "hard_walls" and self._predator_wall_penalty.item() > 0.0:
            rewards_preds -= self._predator_wall_penalty * self._wall_pressure(pred_pos)

        return rewards, rewards_preds, new_deaths, new_potential, pred_phi_bounds

    def _apply_curriculum(self):
        if self.num_predators == 0:
            self.physics.predator_catch_radius = 0.0
            return

        if not self.curriculum_enabled or self.predator_catch_radius_decay_steps <= 0:
            self.physics.predator_catch_radius = self.predator_catch_radius_start
            return

        progress = min(1.0, self.env_step_counter / float(self.predator_catch_radius_decay_steps))
        radius_delta = self.predator_catch_radius_end - self.predator_catch_radius_start
        self.physics.predator_catch_radius = self.predator_catch_radius_start + (radius_delta * progress)
