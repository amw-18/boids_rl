import torch

class BoidsPhysics:
    """
    Vectorized PyTorch implementation of 3D Boids (Starlings) physics.
    Designed for high performance on Apple Silicon (MPS).
    """
    def __init__(
        self,
        num_boids: int,
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
        
        # 'VANTAGE', 'HUNTING', 'DIVING'
        self.predator_state = 'VANTAGE'
        self.predator_timer = 0
        self.predator_target_pos = None

        self.reset()
        
    def reset(self):
        """Randomly initialize positions and velocities."""
        # Positions in [0, space_size]
        self.positions = torch.rand((self.num_boids, 3), device=self.device, dtype=torch.float32) * self.space_size
        # Velocities in random directions, fixed to base_speed
        rand_vel = (torch.rand((self.num_boids, 3), device=self.device, dtype=torch.float32) * 2 - 1)
        speeds = torch.norm(rand_vel, dim=-1, keepdim=True).clamp(min=1e-5)
        self.velocities = (rand_vel / speeds) * self.base_speed
        
        # Initialize Predator State
        # Start predator randomly on the boundary
        predator_pos = torch.rand((1, 3), device=self.device, dtype=torch.float32) * self.space_size
        axis = torch.randint(0, 3, (1,)).item()
        predator_pos[0, axis] = 0.0 if torch.rand(1).item() > 0.5 else self.space_size
        self.predator_position = predator_pos
        
        pred_vel = (torch.rand((1, 3), device=self.device, dtype=torch.float32) * 2 - 1)
        self.predator_velocity = (pred_vel / torch.norm(pred_vel).clamp(min=1e-5)) * self.predator_speed
        
        # Reset State Machine
        self.predator_state = 'VANTAGE'
        self.predator_timer = self.predator_reset_duration
        self.predator_target_pos = None

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
        
        # --- STATE: VANTAGE (Resetting) ---
        if self.predator_state == 'VANTAGE':
            self.predator_timer -= 1
            if self.predator_timer > 0:
                # Climb to vantage altitude
                target_position = torch.tensor([[self.space_size/2, self.space_size/2, self.predator_vantage_z]], device=self.device)
                self.predator_target_pos = target_position
            else:
                # Timer expired, transition to hunting/diving
                self.predator_state = 'EVALUATE'
                
        # --- PREPARE: Density Scanning ---
        # We need density info for both HUNTING and DIVING
        if self.predator_state != 'VANTAGE':
            # Compute pairwise distances between all alive boids
            dist_matrix = torch.cdist(alive_positions, alive_positions)
            
            # Count neighbors within perception radius (subtract 1 so we don't count self)
            neighbors = (dist_matrix < self.perception_radius).sum(dim=1) - 1
            
            # Find isolated boids
            isolated_mask = neighbors < self.min_flock_size
            
            if isolated_mask.any():
                self.predator_state = 'HUNTING'
            else:
                self.predator_state = 'DIVING'
                
        # --- STATE: HUNTING ---
        if self.predator_state == 'HUNTING':
            isolated_positions = alive_positions[isolated_mask]
            
            # Find the closest isolated boid
            dist_to_predator = torch.norm(isolated_positions - self.predator_position, dim=-1)
            closest_idx = torch.argmin(dist_to_predator)
            
            target_position = isolated_positions[closest_idx].unsqueeze(0)
            self.predator_target_pos = target_position
            
        # --- STATE: DIVING ---
        if self.predator_state == 'DIVING':
            # No isolated boids. Dive through the center of the nearest flock.
            # 1. Find nearest boid overall
            dist_to_all = torch.norm(alive_positions - self.predator_position, dim=-1)
            nearest_overall_idx = torch.argmin(dist_to_all)
            
            # 2. Find the local flock of that nearest boid
            nearest_boid_pos = alive_positions[nearest_overall_idx:nearest_overall_idx+1]
            flock_distances = torch.cdist(nearest_boid_pos, alive_positions).squeeze(0)
            local_flock_mask = flock_distances < self.perception_radius
            local_flock_positions = alive_positions[local_flock_mask]
            
            # 3. Target the local Center of Mass
            target_position = local_flock_positions.mean(dim=0, keepdim=True)
            self.predator_target_pos = target_position
            
            # Check if dive is complete (passed through the flock target Z)
            if self.predator_position[0, 2] < target_position[0, 2] + 5.0 and self.predator_velocity[0, 2] < 0:
                self.predator_state = 'VANTAGE'
                self.predator_timer = self.predator_reset_duration
                
        # 2. Seek Behavior (Steering within cone constraint)
        # Use whatever target_position the state machine selected
        target_position = self.predator_target_pos
        desired_velocity = target_position - self.predator_position
        dist_to_target = torch.norm(desired_velocity).clamp(min=1e-5)
        desired_heading = desired_velocity / dist_to_target
        
        current_speed = torch.norm(self.predator_velocity).clamp(min=1e-5)
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
        """Mark boids as dead if the predator touches them."""
        if not self.alive_mask.any():
            return
            
        # Distance from predator to all boids
        dist_to_predator = torch.norm(self.positions - self.predator_position, dim=-1)
        
        # Boids are caught if distance < catch_radius
        caught = dist_to_predator < self.predator_catch_radius
        
        # Update alive mask (dead implies NOT caught previously AND NOT caught now)
        self.alive_mask &= ~caught
