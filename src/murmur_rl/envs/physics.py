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
        self.predator_confusion_dist = 3.0 # Min distance from CoM to be targeted
        
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
        """Update predator velocity and position, targeting the most isolated boid."""
        if not self.alive_mask.any():
            return # Everyone is dead
            
        # 1. Target Selection (Most Isolated = furthest from Center of Mass)
        # Calculate CoM of ALIVE boids
        alive_positions = self.positions[self.alive_mask]
        com = alive_positions.mean(dim=0, keepdim=True)
        
        # Distance of each alive boid to CoM
        dist_to_com = torch.norm(alive_positions - com, dim=-1)
        
        # Find the index of the furthest alive boid
        max_dist_idx = torch.argmax(dist_to_com)
        max_dist = dist_to_com[max_dist_idx]
        
        # If the furthest boid is too close to the flock (confusion effect),
        # the predator loses its lock and just flies towards the center of mass
        if max_dist < self.predator_confusion_dist:
            target_position = com
        else:
            # We need the actual position of the target
            target_position = alive_positions[max_dist_idx].unsqueeze(0)
        
        # 2. Seek Behavior (Steering within cone constraint)
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
