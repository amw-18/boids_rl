import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.agents.starling import StarlingBrain

class RLVision3D:
    def __init__(self, checkpoint_path, num_boids=400, space_size=100.0, device='cpu'):
        self.space_size = space_size
        self.device = torch.device(device)
        
        # Initialize VectorEnv
        self.env = VectorMurmurationEnv(
            num_agents=num_boids,
            space_size=space_size,
            perception_radius=15.0,
            device=device
        )
        self.env.physics.base_speed = 5.0
        self.env.physics.max_turn_angle = 0.5
        self.env.physics.max_force = 2.0
        
        # Initialize RL Brain
        self.brain = StarlingBrain(obs_dim=16, action_dim=3, hidden_size=64).to(self.device)
        self.brain.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.brain.eval()
        
        self.obs = self.env.reset()
        
        # Matplotlib Setup
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0, space_size])
        self.ax.set_ylim([0, space_size])
        self.ax.set_zlim([0, space_size])
        self.ax.set_axis_off()  # Hide axes
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Initialize Boid scatter
        positions = self.env.physics.positions.cpu().numpy()
        self.scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c='cyan', marker='^', s=10, alpha=0.8
        )
        
        # Initialize Predator scatter (Red)
        pred_pos = self.env.physics.predator_position.cpu().numpy()
        self.predator_scatter = self.ax.scatter(
            pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
            c='red', marker='o', s=40, alpha=1.0
        )
        
    def update(self, frame):
        with torch.no_grad():
            action_mean, action_std, _ = self.brain(self.obs)
            # Use deterministic actions for visualization
            action = action_mean 
            
        self.obs, rewards, dones = self.env.step(action)
        
        # Update Boid scatter plot data (only alive)
        alive = self.env.physics.alive_mask.cpu().numpy()
        positions = self.env.physics.positions.cpu().numpy()[alive]
        
        if len(positions) > 0:
            self.scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
        # Update Predator scatter data
        pred_pos = self.env.physics.predator_position.cpu().numpy()
        self.predator_scatter._offsets3d = (pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2])
        
        return self.scatter, self.predator_scatter

    def animate(self, frames=500, interval=30, save_path=None):
        anim = animation.FuncAnimation(
            self.fig, self.update, frames=frames, interval=interval, blit=False
        )
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints1/starling_brain_ep5000.pth")
    parser.add_argument("--save", type=str, default="murmuration_rl.gif")
    parser.add_argument("--frames", type=int, default=3600)
    args = parser.parse_args()
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating Simulation using {device_name}...")
    vis = RLVision3D(checkpoint_path=args.checkpoint, num_boids=400, space_size=100.0, device=device_name)
    vis.animate(frames=args.frames, save_path=args.save)
