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
        
        # Vectorized Predator States: 0=VANTAGE, 1=HUNTING, 2=DIVING
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
            actions: Tensor of shape (num_boids, 3) representing external forces / RL agent actions.
                     This is the primary steering force.
        """
        # BOUNDARY AVOIDANCE (Soft boundaries)
        # We still want soft boundaries so the boids don't fly off to infinity
        boundary_push = torch.zeros_like(self.positions)
        margin = self.space_size * 0.1  # 10% margin
        
        # Lower bounds
        lower_mask = self.positions < margin
        boundary_push += lower_mask.float() * (margin - self.positions)
        
        # Upper bounds
        upper_mask = self.positions > (self.space_size - margin)
        boundary_push -= upper_mask.float() * (self.positions - (self.space_size - margin))
        
        # Combine forces
        total_force = self.boundary_weight * boundary_push
        
        # Apply external/RL actions if provided
        if actions is not None:
            # We assume actions are additional force vectors provided by the NN
            total_force += actions
            
        # Limit total external force per timestep
        force_magnitudes = torch.norm(total_force, dim=-1, keepdim=True).clamp(min=1e-5)
        total_force = torch.where(
            force_magnitudes > self.max_force,
            (total_force / force_magnitudes) * self.max_force,
            total_force
        )
        
        # Current heading
        speed = torch.norm(self.velocities, dim=-1, keepdim=True).clamp(min=1e-5)
        current_heading = self.velocities / speed
        
        # Desired heading based on applied steering forces
        new_v = self.velocities + total_force * self.dt
        new_speed = torch.norm(new_v, dim=-1, keepdim=True).clamp(min=1e-5)
        desired_heading = new_v / new_speed
        
        # Compute angle between current taking and desired heading
        dot = (current_heading * desired_heading).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        
        # Orthogonal projection for rotation plane
        w = desired_heading - dot * current_heading
        w_norm = torch.norm(w, dim=-1, keepdim=True).clamp(min=1e-5)
        w_unit = w / w_norm
        
        # Clamp turn angle to the steering cone (max_turn_angle)
        turn_angle = torch.clamp(angle, max=self.max_turn_angle)
        
        # Spherical linear interpolation towards desired heading
        final_heading = torch.cos(turn_angle) * current_heading + torch.sin(turn_angle) * w_unit
        
        # Safeguard against tiny angles causing numerical instability
        final_heading = torch.where(angle < 1e-4, desired_heading, final_heading)
        final_heading = final_heading / torch.norm(final_heading, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Enforce exactly constant base speed
        self.velocities = final_heading * self.base_speed
        
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
        self.predator_timer = torch.where(is_vantage, self.predator_timer - 1, self.predator_timer)
        
        climbing_mask = is_vantage & (self.predator_timer > 0)
        if climbing_mask.any():
            self.predator_target_pos[climbing_mask, 0] = self.space_size / 2.0
            self.predator_target_pos[climbing_mask, 1] = self.space_size / 2.0
            self.predator_target_pos[climbing_mask, 2] = self.predator_vantage_z
            
        expired_vantage_mask = is_vantage & (self.predator_timer <= 0)
        
        # Evaluate Hunting vs Diving for predators ready to strike or already striking
        eval_mask = expired_vantage_mask | (self.predator_state != 0)
        if eval_mask.any():
            dist_matrix = torch.cdist(alive_positions, alive_positions)
            neighbors = (dist_matrix < self.perception_radius).sum(dim=1) - 1
            isolated_mask = neighbors < self.min_flock_size
            has_isolated = isolated_mask.any()
            
            if has_isolated:
                self.predator_state[eval_mask] = 1 # HUNTING
            else:
                self.predator_state[eval_mask] = 2 # DIVING

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
        
        # 3. Apply physics at fixed predator speed
        self.predator_velocity = final_heading * self.predator_speed
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
