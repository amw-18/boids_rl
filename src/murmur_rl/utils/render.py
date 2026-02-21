import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from murmur_rl.envs.physics import BoidsPhysics

class BoidsVisualizer3D:
    def __init__(self, num_boids=500, space_size=100.0, device='cpu'):
        self.space_size = space_size
        self.device = torch.device('mps' if torch.backends.mps.is_available() and device == 'mps' else device)
        
        # We initialize with unbiased physics. 
        # (Be aware: without an RL policy acting on it, it will just be random motion + boundary avoidance + predator chasing)
        self.physics = BoidsPhysics(
            num_boids=num_boids,
            space_size=space_size,
            device=self.device,
            perception_radius=15.0,
            max_speed=2.0,
            max_force=0.1
        )
        
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0, space_size])
        self.ax.set_ylim([0, space_size])
        self.ax.set_zlim([0, space_size])
        self.ax.set_axis_off()  # Hide axes for better visual

        # Set background to black
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Initialize scatter plot
        # Starlings are often dark/iridescent. We'll use a bright color for visibility on black.
        positions = self.physics.positions.cpu().numpy()
        self.scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c='cyan', marker='^', s=10, alpha=0.8
        )
        
        # Initialize Predator scatter (Red, slightly larger)
        pred_pos = self.physics.predator_position.cpu().numpy()
        self.predator_scatter = self.ax.scatter(
            pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
            c='red', marker='o', s=40, alpha=1.0
        )
        
    def update(self, frame):
        self.physics.step()
        
        # Update Boid scatter plot data
        # Only plot alive boids
        alive = self.physics.alive_mask.cpu().numpy()
        positions = self.physics.positions.cpu().numpy()[alive]
        if len(positions) > 0:
            self.scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
        # Update Predator scatter data
        pred_pos = self.physics.predator_position.cpu().numpy()
        self.predator_scatter._offsets3d = (pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2])
        
        return self.scatter, self.predator_scatter

    def animate(self, frames=500, interval=20, save_path=None):
        anim = animation.FuncAnimation(
            self.fig, self.update, frames=frames, interval=interval, blit=False
        )
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    print("Starting 3D Murmuration Visualization...")
    # Use MPS if available for fast simulation while rendering on CPU
    vis = BoidsVisualizer3D(num_boids=800, space_size=100.0, device='mps')
    vis.animate(save_path='/Users/amwal/.gemini/antigravity/brain/7baa8ad9-0b0c-4cb4-8636-750cf506fa31/test.gif')
