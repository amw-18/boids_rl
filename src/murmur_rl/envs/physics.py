import torch

class BoidsPhysics:
    """
    Vectorized PyTorch implementation of 3D Boids (Starlings) physics.
    Designed for high performance on Apple Silicon (MPS).
    """
    def __init__(
        self,
        *,
        num_boids: int,
        num_predators: int = 1,
        space_size: float = 100.0,
        device: torch.device = torch.device('cpu'),
        perception_radius: float = 10.0,
        base_speed: float = 5.0,
        max_turn_angle: float = 0.5, # radians per timestep
        max_force: float = 2.0,
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
        
        # Predator properties
        self.predator_base_speed = base_speed
        self.predator_sprint_speed = base_speed * 1.5 
        self.predator_turn_angle = max_turn_angle * 1.5
        self.predator_catch_radius = 2.0
        
        # Co-Evolution Parameters: Stamina Economy
        self.predator_max_stamina = 100.0 # total sprint capacity
        self.predator_sprint_drain = 1.0  # Stamina drain per frame while sprinting
        self.predator_recovery_rate = 0.5 # Stamina recovery per frame while cruising
        
        # Track individual predator energy
        self.predator_stamina = torch.full((self.num_predators,), self.predator_max_stamina, device=self.device)
        self.predator_cooldown = torch.zeros(self.num_predators, dtype=torch.long, device=self.device)
        self.predator_cooldown_duration = 50 # Frames disabled after a successful catch
        self.predator_time_since_cooldown = torch.zeros(self.num_predators, dtype=torch.long, device=self.device)


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
        predator_pos = torch.rand((self.num_predators, 3), device=self.device, dtype=torch.float32) * self.space_size
        # Start predators randomly on the boundary
        # axis = torch.randint(0, 3, (self.num_predators,))
        # for i in range(self.num_predators):
        #     predator_pos[i, axis[i]] = 0.0 if torch.rand(1).item() > 0.5 else self.space_size
        self.predator_position = predator_pos
        
        pred_vel = (torch.rand((self.num_predators, 3), device=self.device, dtype=torch.float32) * 2 - 1)
        speeds_pred = torch.norm(pred_vel, dim=-1, keepdim=True).clamp(min=1e-5)
        self.predator_velocity = (pred_vel / speeds_pred) * self.predator_base_speed
        
        # Initialize predator up vectors for 6-DOF tracking
        pred_z_axis = torch.zeros_like(self.predator_velocity)
        pred_z_axis[:, 2] = 1.0
        pred_z_axis = torch.where(torch.abs(self.predator_velocity[:, 2:3] / self.predator_base_speed) > 0.99, torch.tensor([0.0, 1.0, 0.0], device=self.device), pred_z_axis)
        
        pred_right = torch.cross(self.predator_velocity / self.predator_base_speed, pred_z_axis, dim=-1)
        pred_right = pred_right / torch.norm(pred_right, dim=-1, keepdim=True).clamp(min=1e-5)
        self.predator_up_vectors = torch.cross(pred_right, self.predator_velocity / self.predator_base_speed, dim=-1)
        self.predator_up_vectors = self.predator_up_vectors / torch.norm(self.predator_up_vectors, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Reset Co-Evolution state
        self.predator_stamina.fill_(self.predator_max_stamina)
        self.predator_cooldown.zero_()
        self.predator_time_since_cooldown.zero_()

        # Boids that have been eaten (boolean mask, True = alive, False = dead)
        self.alive_mask = torch.ones(self.num_boids, dtype=torch.bool, device=self.device)
        
    def step(self, boid_actions: torch.Tensor = None, predator_actions: torch.Tensor = None):
        """
        Advance the physics by one timestep.
        
        Args:
            boid_actions: Tensor of shape (num_boids, 3) representing (Thrust, Roll Rate, Pitch Rate) in [-1, 1].
            predator_actions: Tensor of shape (num_predators, 3) representing (Sprint, Roll Rate, Pitch Rate) in [-1, 1].
        """
        if boid_actions is None:
            boid_actions = torch.zeros((self.num_boids, 3), device=self.device)
            
        thrust_action = boid_actions[:, 0:1] # [-1, 1]
        roll_action = boid_actions[:, 1:2]   # [-1, 1]
        pitch_action = boid_actions[:, 2:3]  # [-1, 1]
        
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
        
        # Aerodynamic Limits: Cap speed between 0.5 and base_speed
        new_speed = torch.clamp(new_speed, min=0.5, max=self.base_speed)
        
        self.velocities = forward_new * new_speed
        
        # Kill velocities of dead boids so they freeze/fall out of sim
        self.velocities[~self.alive_mask] = 0.0
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # === PREDATOR PHYSICS ===
        self._update_predator(predator_actions)
        self._check_captures()
        
    def _update_predator(self, actions: torch.Tensor = None):
        """Update predator velocity and position via continuous RL controls with stamina."""
        if actions is None:
            actions = torch.zeros((self.num_predators, 3), device=self.device)

        sprint_action = actions[:, 0:1] # [-1, 1], >0 is sprinting
        roll_action = actions[:, 1:2]   # [-1, 1]
        pitch_action = actions[:, 2:3]  # [-1, 1]
        
        roll_angle = roll_action * self.predator_turn_angle
        pitch_angle = pitch_action * self.predator_turn_angle
        
        # 1. Orientation Update (Identical to Boids)
        speed = torch.norm(self.predator_velocity, dim=-1, keepdim=True).clamp(min=1e-5)
        forward = self.predator_velocity / speed
        up = self.predator_up_vectors
        right = torch.cross(forward, up, dim=-1)
        right = right / torch.norm(right, dim=-1, keepdim=True).clamp(min=1e-5)
        
        cos_r = torch.cos(roll_angle)
        sin_r = torch.sin(roll_angle)
        up_rolled = up * cos_r + right * sin_r
        
        cos_p = torch.cos(pitch_angle)
        sin_p = torch.sin(pitch_angle)
        forward_new = forward * cos_p + up_rolled * sin_p
        up_new = up_rolled * cos_p - forward * sin_p
        
        forward_new = forward_new / torch.norm(forward_new, dim=-1, keepdim=True).clamp(min=1e-5)
        up_new = up_new / torch.norm(up_new, dim=-1, keepdim=True).clamp(min=1e-5)
        
        right_new = torch.cross(forward_new, up_new, dim=-1)
        right_new = right_new / torch.norm(right_new, dim=-1, keepdim=True).clamp(min=1e-5)
        self.predator_up_vectors = torch.cross(right_new, forward_new, dim=-1)
        self.predator_up_vectors = self.predator_up_vectors / torch.norm(self.predator_up_vectors, dim=-1, keepdim=True).clamp(min=1e-5)
        
        # 2. Stamina Management & Speed Calculation
        # Decrement cooldown timers
        self.predator_cooldown = torch.clamp(self.predator_cooldown - 1, min=0)
        is_cooldown = self.predator_cooldown > 0
        
        self.predator_time_since_cooldown = torch.where(
            ~is_cooldown,
            self.predator_time_since_cooldown + 1,
            torch.zeros_like(self.predator_time_since_cooldown)
        )
        
        # Intent to sprint requires stamina > 0 and no cooldown
        is_sprinting = (sprint_action.squeeze(-1) > 0) & (self.predator_stamina > 0) & ~is_cooldown
        is_cruising = ~is_sprinting
        
        # Update stamina
        self.predator_stamina = torch.where(
            is_sprinting,
            self.predator_stamina - self.predator_sprint_drain,
            self.predator_stamina + self.predator_recovery_rate
        ).clamp(min=0.0, max=self.predator_max_stamina)
        
        # Assign speed based on state
        target_speed = torch.full((self.num_predators, 1), self.predator_base_speed, device=self.device)
        target_speed[is_sprinting] = self.predator_sprint_speed
        # During cooldown after a catch, they are slow
        target_speed[is_cooldown] = self.predator_base_speed * 0.5
        
        # Inertia blending for smooth speed transitions
        new_speed = speed * 0.9 + target_speed * 0.1
        
        self.predator_velocity = forward_new * new_speed
        self.predator_position += self.predator_velocity * self.dt
        
    def _check_captures(self):
        """Mark boids as dead if ANY predator touches them. Apply cooldowns."""
        if not self.alive_mask.any():
            return
            
        # Distance from ALL predators to ALL boids
        dist_to_predator = torch.cdist(self.predator_position, self.positions)
        
        # Boids are caught if distance < catch_radius for ANY predator
        caught_matrix = dist_to_predator < self.predator_catch_radius
        caught = caught_matrix.any(dim=0) # (N_boids,)
        
        # Update alive mask
        self.alive_mask &= ~caught
        
        # WHICH predators made a catch?
        caught_by_predator = caught_matrix.any(dim=1) # (num_predators,)
        
        if caught_by_predator.any():
            # Apply cooldown to successful predators, freezing their ability to sprint
            # and slowing them down for the duration.
            self.predator_cooldown[caught_by_predator] = self.predator_cooldown_duration
            self.predator_time_since_cooldown[caught_by_predator] = 0
