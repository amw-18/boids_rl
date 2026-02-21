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
        max_speed: float = 5.0,
        max_force: float = 0.5,
        boundary_weight: float = 2.0,
        dt: float = 0.1
    ):
        self.num_boids = num_boids
        self.space_size = space_size
        self.device = device
        
        # Physics hyperparams
        self.perception_radius = perception_radius
        self.max_speed = max_speed
        self.max_force = max_force
        self.dt = dt
        
        # Rule weights (Soft boundary only, no hardcoded flocking)
        self.boundary_weight = boundary_weight
        
        # Predator features
        self.predator_speed = max_speed * 1.5 
        self.predator_max_force = max_force * 2.0
        self.predator_catch_radius = 2.0
        
        self.reset()
        
    def reset(self):
        """Randomly initialize positions and velocities."""
        # Positions in [0, space_size]
        self.positions = torch.rand((self.num_boids, 3), device=self.device, dtype=torch.float32) * self.space_size
        # Velocities in [-max_speed, max_speed]
        self.velocities = (torch.rand((self.num_boids, 3), device=self.device, dtype=torch.float32) * 2 - 1) * self.max_speed
        
        # Initialize Predator State
        # Start predator randomly on the boundary
        predator_pos = torch.rand((1, 3), device=self.device, dtype=torch.float32) * self.space_size
        axis = torch.randint(0, 3, (1,)).item()
        predator_pos[0, axis] = 0.0 if torch.rand(1).item() > 0.5 else self.space_size
        self.predator_position = predator_pos
        self.predator_velocity = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        
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
            
        # Limit total force
        force_magnitudes = torch.norm(total_force, dim=-1, keepdim=True).clamp(min=1e-5)
        total_force = torch.where(
            force_magnitudes > self.max_force,
            total_force / force_magnitudes * self.max_force,
            total_force
        )
        
        # Update velocities
        self.velocities += total_force * self.dt
        
        # Limit speeds
        speed = torch.norm(self.velocities, dim=-1, keepdim=True).clamp(min=1e-5)
        self.velocities = torch.where(
            speed > self.max_speed,
            self.velocities / speed * self.max_speed,
            self.velocities
        )
        
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
        target_idx_in_alive = torch.argmax(dist_to_com)
        # We need the actual position of the target
        target_position = alive_positions[target_idx_in_alive].unsqueeze(0)
        
        # 2. Seek Behavior (Steer towards target)
        desired_velocity = target_position - self.predator_position
        dist_to_target = torch.norm(desired_velocity).clamp(min=1e-5)
        
        # Normalize and scale to max speed
        desired_velocity = (desired_velocity / dist_to_target) * self.predator_speed
        
        # Steering = Desired - Current
        steer = desired_velocity - self.predator_velocity
        steer_mag = torch.norm(steer).clamp(min=1e-5)
        if steer_mag > self.predator_max_force:
            steer = (steer / steer_mag) * self.predator_max_force
            
        # 3. Apply physics
        self.predator_velocity += steer * self.dt
        
        predator_speed = torch.norm(self.predator_velocity).clamp(min=1e-5)
        if predator_speed > self.predator_speed:
            self.predator_velocity = (self.predator_velocity / predator_speed) * self.predator_speed
            
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
