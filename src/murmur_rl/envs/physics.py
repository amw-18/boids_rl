import torch

class BoidsPhysics:
    """
    Vectorized PyTorch implementation of 3D Boids (Starlings) physics.
    Designed for high performance on Apple Silicon (MPS).
    """
    def __init__(
        self,
        num_boids: int,
        num_predators: int = 1,
        space_size: float = 100.0,
        device: torch.device = torch.device('cpu'),
        perception_radius: float = 10.0,
        base_speed: float = 5.0,
        max_turn_angle: float = 0.5, # radians per timestep
        max_force: float = 2.0,
        boundary_weight: float = 2.0,
        dt: float = 0.1
    ):
        self.num_boids = num_boids
        self.num_predators = num_predators
        self.space_size = space_size
        self.device = device
        
        # Physics hyperparams
        self.perception_radius = perception_radius
        self.base_speed = base_speed
        self.max_turn_angle = max_turn_angle
        self.max_force = max_force
        self.dt = dt
        
        # Rule weights (Soft boundary only, no hardcoded flocking)
        self.boundary_weight = boundary_weight
        
        # Predator features
        self.predator_speed = base_speed * 1.5 
        self.predator_turn_angle = max_turn_angle * 1.5
        self.predator_catch_radius = 2.0
        
        # Falcon State Machine Parameters
        self.min_flock_size = 5
        self.predator_vantage_z = space_size * 0.9
        self.predator_reset_duration = 50 # Timesteps to stay in VANTAGE
        self.predator_loiter_base = 100 # Base LOITER duration
        self.predator_loiter_variance = 300 # High variance added to base
        
        # Vectorized Predator States: 0=VANTAGE, 1=HUNTING, 2=DIVING, 3=LOITER
        self.predator_state = torch.zeros(self.num_predators, dtype=torch.long, device=self.device)
        self.predator_timer = torch.zeros(self.num_predators, dtype=torch.long, device=self.device)
        self.predator_target_pos = torch.zeros((self.num_predators, 3), device=self.device)


        self.reset()
        
    def reset(self):
        """Randomly initialize positions and velocities."""
        # Implement clustered spawn logic for boids to prevent Perception Starvation
        # Boids spawn uniformly within a 100.0 center box (or space_size if space_size < 100.0)
        spawn_box_size = min(100.0, self.space_size)
        spawn_center = self.space_size / 2.0
        min_bound = spawn_center - (spawn_box_size / 2.0)
        
        # Positions in clustered spawn block
        self.positions = min_bound + torch.rand((self.num_boids, 3), device=self.device, dtype=torch.float32) * spawn_box_size
        
        # Velocities in random directions, fixed to base_speed
        rand_vel = (torch.rand((self.num_boids, 3), device=self.device, dtype=torch.float32) * 2 - 1)
        speeds = torch.norm(rand_vel, dim=-1, keepdim=True).clamp(min=1e-5)
        self.velocities = (rand_vel / speeds) * self.base_speed
        
        # Initialize up vectors (orthogonal to velocities)
        z_axis = torch.zeros_like(self.velocities)
        z_axis[:, 2] = 1.0
        # If velocity is exactly along Z, use Y-axis as base
        z_axis = torch.where(torch.abs(self.velocities[:, 2:3] / self.base_speed) > 0.99, torch.tensor([0.0, 1.0, 0.0], device=self.device), z_axis)
        
        right = torch.cross(self.velocities / self.base_speed, z_axis, dim=-1)
        right = right / torch.norm(right, dim=-1, keepdim=True).clamp(min=1e-5)
        self.up_vectors = torch.cross(right, self.velocities / self.base_speed, dim=-1)
        self.up_vectors = self.up_vectors / torch.norm(self.up_vectors, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Initialize Predator State
        # Start predators randomly on the boundary
        predator_pos = torch.rand((self.num_predators, 3), device=self.device, dtype=torch.float32) * self.space_size
        axis = torch.randint(0, 3, (self.num_predators,))
        for i in range(self.num_predators):
            predator_pos[i, axis[i]] = 0.0 if torch.rand(1).item() > 0.5 else self.space_size
        self.predator_position = predator_pos
        
        pred_vel = (torch.rand((self.num_predators, 3), device=self.device, dtype=torch.float32) * 2 - 1)
        speeds_pred = torch.norm(pred_vel, dim=-1, keepdim=True).clamp(min=1e-5)
        self.predator_velocity = (pred_vel / speeds_pred) * self.predator_speed
        
        # Reset State Machine
        self.predator_state.zero_() # VANTAGE
        self.predator_timer.fill_(self.predator_reset_duration)
        self.predator_target_pos.zero_()

        # Boids that have been eaten (boolean mask, True = alive, False = dead)
        self.alive_mask = torch.ones(self.num_boids, dtype=torch.bool, device=self.device)
        
    def step(self, actions: torch.Tensor = None):
        """
        Advance the physics by one timestep.
        
        Args:
            actions: Tensor of shape (num_boids, 3) representing (Thrust, Roll Rate, Pitch Rate) in [-1, 1].
        """
        if actions is None:
            actions = torch.zeros((self.num_boids, 3), device=self.device)
            
        thrust_action = actions[:, 0:1] # [-1, 1]
        roll_action = actions[:, 1:2]   # [-1, 1]
        pitch_action = actions[:, 2:3]  # [-1, 1]
        
        # Apply scaling based on physics limits
        thrust = thrust_action * self.max_force
        roll_angle = roll_action * self.max_turn_angle
        pitch_angle = pitch_action * self.max_turn_angle
        
        # 1. Orientation
        speed = torch.norm(self.velocities, dim=-1, keepdim=True).clamp(min=1e-5)
        forward = self.velocities / speed
        up = self.up_vectors
        # Right vector (Orthogonal to Forward and Up)
        right = torch.cross(forward, up, dim=-1)
        right = right / torch.norm(right, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Roll: Rotate Up and Right around Forward
        cos_r = torch.cos(roll_angle)
        sin_r = torch.sin(roll_angle)
        
        up_rolled = up * cos_r + right * sin_r
        
        # Pitch: Rotate Forward and Up_rolled around Right_rolled
        cos_p = torch.cos(pitch_angle)
        sin_p = torch.sin(pitch_angle)
        
        forward_new = forward * cos_p + up_rolled * sin_p
        up_new = up_rolled * cos_p - forward * sin_p
        
        # Normalize to prevent numerical drift
        forward_new = forward_new / torch.norm(forward_new, dim=-1, keepdim=True).clamp(min=1e-5)
        up_new = up_new / torch.norm(up_new, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Re-enforce strict orthogonality for the next frame
        right_new = torch.cross(forward_new, up_new, dim=-1)
        right_new = right_new / torch.norm(right_new, dim=-1, keepdim=True).clamp(min=1e-5)
        self.up_vectors = torch.cross(right_new, forward_new, dim=-1)
        self.up_vectors = self.up_vectors / torch.norm(self.up_vectors, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # 2. Velocity & Thrust
        new_speed = speed + thrust * self.dt
        
        # Aerodynamic Limits: Cap speed between 0.5 and 10.0
        new_speed = torch.clamp(new_speed, min=0.5, max=10.0)
        
        self.velocities = forward_new * new_speed
        
        # Kill velocities of dead boids so they freeze/fall out of sim
        self.velocities[~self.alive_mask] = 0.0
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # === PREDATOR PHYSICS ===
        self._update_predator()
        self._check_captures()
        
    def _update_predator(self):
        """Update predator velocity and position via the Falcon State Machine."""
        if not self.alive_mask.any():
            return # Everyone is dead
            
        alive_positions = self.positions[self.alive_mask]
        
        # Vectorized State Evaluation
        is_vantage = self.predator_state == 0
        
        # 1. Height-based Vantage Check:
        # A predator is "climbing" if it is in VANTAGE state and its Z-position is below the vantage height.
        climbing_mask = is_vantage & (self.predator_position[:, 2] < self.predator_vantage_z - 5.0)
        
        if climbing_mask.any():
            num_climbing = climbing_mask.sum()
            # Spread predators out slightly around the center
            spread = self.space_size * 0.1
            x_offsets = (torch.rand(num_climbing, device=self.device) * 2 - 1) * spread
            y_offsets = (torch.rand(num_climbing, device=self.device) * 2 - 1) * spread
            
            self.predator_target_pos[climbing_mask, 0] = (self.space_size / 2.0) + x_offsets
            self.predator_target_pos[climbing_mask, 1] = (self.space_size / 2.0) + y_offsets
            self.predator_target_pos[climbing_mask, 2] = self.predator_vantage_z
            
        # A predator has reached the vantage point when its Z-position is high enough
        reached_vantage_mask = is_vantage & (self.predator_position[:, 2] >= self.predator_vantage_z - 5.0)
        
        # We also enforce the minimum wait time here so they don't strike immediately upon reaching the height
        self.predator_timer = torch.where(reached_vantage_mask & (self.predator_timer > 0), self.predator_timer - 1, self.predator_timer)
        
        expired_vantage_mask = reached_vantage_mask & (self.predator_timer <= 0)
        
        # Transition from VANTAGE to LOITER
        if expired_vantage_mask.any():
            self.predator_state[expired_vantage_mask] = 3 # LOITER
            num_loitering_new = expired_vantage_mask.sum()
            # High variance distribution for LOITER duration
            random_duration = self.predator_loiter_base + torch.randint(0, self.predator_loiter_variance, (num_loitering_new,), device=self.device)
            self.predator_timer[expired_vantage_mask] = random_duration
            
        # 2. State Evaluation (LOITER)
        is_loitering = self.predator_state == 3
        if is_loitering.any():
            self.predator_timer[is_loitering] -= 1
            
            # While loitering, randomly wander above vantage height
            # We pick a new random nearby target periodically
            needs_new_target = is_loitering & (torch.rand(self.num_predators, device=self.device) < 0.05)
            if needs_new_target.any():
                num_new_targets = needs_new_target.sum()
                spread = self.space_size * 0.2
                x_offsets = (torch.rand(num_new_targets, device=self.device) * 2 - 1) * spread
                y_offsets = (torch.rand(num_new_targets, device=self.device) * 2 - 1) * spread
                z_offsets = torch.rand(num_new_targets, device=self.device) * (self.space_size - self.predator_vantage_z)
                
                self.predator_target_pos[needs_new_target, 0] = (self.space_size / 2.0) + x_offsets
                self.predator_target_pos[needs_new_target, 1] = (self.space_size / 2.0) + y_offsets
                self.predator_target_pos[needs_new_target, 2] = self.predator_vantage_z + z_offsets

        expired_loiter_mask = is_loitering & (self.predator_timer <= 0)
        
        # Evaluate Hunting vs Diving for predators ready to strike or already striking
        # They can only enter these states if they have finished waiting at the vantage point.
        eval_mask = expired_loiter_mask | (self.predator_state == 1) | (self.predator_state == 2)
        if eval_mask.any():
            dist_matrix = torch.cdist(alive_positions, alive_positions)
            neighbors = (dist_matrix < self.perception_radius).sum(dim=1) - 1
            isolated_mask = neighbors < self.min_flock_size
            has_isolated = isolated_mask.any()
            
            # Specifically for newly expired loiterers, choose their next state
            if expired_loiter_mask.any():
                self.predator_state[expired_loiter_mask] = 1 if has_isolated else 2

        # Process HUNTING
        is_hunting = self.predator_state == 1
        if is_hunting.any():
            isolated_positions = alive_positions[isolated_mask]
            hunting_preds = self.predator_position[is_hunting]
            dist_to_isolated = torch.cdist(hunting_preds, isolated_positions)
            closest_idx = torch.argmin(dist_to_isolated, dim=1)
            self.predator_target_pos[is_hunting] = isolated_positions[closest_idx]
            
        # Process DIVING
        is_diving = self.predator_state == 2
        if is_diving.any():
            diving_preds = self.predator_position[is_diving]
            dist_to_all = torch.cdist(diving_preds, alive_positions)
            nearest_overall_idx = torch.argmin(dist_to_all, dim=1)
            
            for i, idx in enumerate(torch.where(is_diving)[0]):
                nearest_boid_pos = alive_positions[nearest_overall_idx[i]:nearest_overall_idx[i]+1]
                flock_distances = torch.cdist(nearest_boid_pos, alive_positions).squeeze(0)
                local_flock_mask = flock_distances < self.perception_radius
                local_flock_positions = alive_positions[local_flock_mask]
                self.predator_target_pos[idx] = local_flock_positions.mean(dim=0)
                
            # Check Dive Completion
            target_zs = self.predator_target_pos[is_diving, 2]
            pred_zs = self.predator_position[is_diving, 2]
            pred_vz = self.predator_velocity[is_diving, 2]
            
            completed_dives = (pred_zs < target_zs + 5.0) & (pred_vz < 0)
            if completed_dives.any():
                diving_indices = torch.where(is_diving)[0]
                completed_indices = diving_indices[completed_dives]
                self.predator_state[completed_indices] = 0 # VANTAGE
                self.predator_timer[completed_indices] = self.predator_reset_duration

        # 2. Seek Behavior (Steering within cone constraint)
        desired_velocity = self.predator_target_pos - self.predator_position
        dist_to_target = torch.norm(desired_velocity, dim=-1, keepdim=True).clamp(min=1e-5)
        desired_heading = desired_velocity / dist_to_target
        
        current_speed = torch.norm(self.predator_velocity, dim=-1, keepdim=True).clamp(min=1e-5)
        current_heading = self.predator_velocity / current_speed
        
        dot = (current_heading * desired_heading).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        
        w = desired_heading - dot * current_heading
        w_norm = torch.norm(w, dim=-1, keepdim=True).clamp(min=1e-5)
        w_unit = w / w_norm
        
        turn_angle = torch.clamp(angle, max=self.predator_turn_angle)
        
        final_heading = torch.cos(turn_angle) * current_heading + torch.sin(turn_angle) * w_unit
        final_heading = torch.where(angle < 1e-4, desired_heading, final_heading)
        final_heading = final_heading / torch.norm(final_heading, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # 3. Apply physics at fixed predator speed, except slower during LOITER
        is_loitering_mask = (self.predator_state == 3).unsqueeze(-1)
        current_target_speed = torch.where(is_loitering_mask, self.predator_speed * 0.3, self.predator_speed)
        
        self.predator_velocity = final_heading * current_target_speed
        self.predator_position += self.predator_velocity * self.dt
        
    def _check_captures(self):
        """Mark boids as dead if ANY predator touches them."""
        if not self.alive_mask.any():
            return
            
        # Distance from ALL predators to ALL boids
        # positions: (N_boids, 3), predator_position: (num_predators, 3)
        # Using cdist -> (num_predators, N_boids)
        dist_to_predator = torch.cdist(self.predator_position, self.positions)
        
        # Boids are caught if distance < catch_radius for ANY predator
        caught_matrix = dist_to_predator < self.predator_catch_radius
        caught = caught_matrix.any(dim=0) # (N_boids,)
        
        # Update alive mask
        self.alive_mask &= ~caught
        
        # If any boids were caught by a specific predator, reset that predator to VANTAGE
        # dist_to_predator is (num_predators, N_boids)
        # caught_matrix is (num_predators, N_boids)
        # We need to know WHICH predator made a catch:
        caught_by_predator = caught_matrix.any(dim=1) # (num_predators,)
        
        if caught_by_predator.any():
            self.predator_state[caught_by_predator] = 0 # VANTAGE
            self.predator_timer[caught_by_predator] = self.predator_reset_duration # Resets the wait timer at the top
            
            num_caught = caught_by_predator.sum()
            spread = self.space_size * 0.1
            x_offsets = (torch.rand(num_caught, device=self.device) * 2 - 1) * spread
            y_offsets = (torch.rand(num_caught, device=self.device) * 2 - 1) * spread
            
            # The predator immediately targets the center/top to climb back up (with some spread)
            self.predator_target_pos[caught_by_predator, 0] = (self.space_size / 2.0) + x_offsets
            self.predator_target_pos[caught_by_predator, 1] = (self.space_size / 2.0) + y_offsets
            self.predator_target_pos[caught_by_predator, 2] = self.predator_vantage_z
