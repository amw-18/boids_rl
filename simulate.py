import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.agents.starling import StarlingBrain

class RLVision3D:
    def __init__(self, checkpoint_path, num_boids=400, num_predators=5, space_size=100.0, device='cpu'):
        self.space_size = space_size
        self.device = torch.device(device)
        
        # Initialize VectorEnv
        self.env = VectorMurmurationEnv(
            num_agents=num_boids,
            num_predators=num_predators,
            space_size=space_size,
            perception_radius=15.0,
            device=device
        )
        self.env.physics.base_speed = 5.0
        self.env.physics.max_turn_angle = 0.5
        self.env.physics.max_force = 2.0
        
        # Load checkpoint directly to find the expected Critic dimensions
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Determine global_obs_dim dynamically so we don't crash when running simulation 
        # with a different number of boids than training
        if 'critic.0.weight' in checkpoint:
            expected_global_obs_dim = checkpoint['critic.0.weight'].shape[1]
        else:
            expected_global_obs_dim = self.env.global_obs_dim  # Fallback
            
        # Initialize RL Brain
        self.brain = StarlingBrain(
            obs_dim=18, 
            global_obs_dim=expected_global_obs_dim, 
            action_dim=3, 
            hidden_size=64
        ).to(self.device)
        
        try:
            self.brain.load_state_dict(checkpoint, strict=False)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n[ERROR] Checkpoint Loading Failed: Size Mismatch")
                print(f"The checkpoint '{checkpoint_path}' appears to be trained on an older version of the environment.")
                print("The observation space has been upgraded from 16 to 18 dimensions (3D boundary relative positions).")
                print("Please train a new agent to use with the current environment.")
                import sys
                sys.exit(1)
            else:
                raise e
                
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
            features = self.brain.actor_feature_extractor(self.obs)
            action_mean = self.brain.actor_mean(features)
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
        import imageio
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        if not save_path:
            save_path = "murmuration_rl.mp4"
            
        # Swap .gif to .mp4 automatically to guarantee streaming
        if save_path.endswith('.gif'):
            save_path = save_path.replace('.gif', '.mp4')
            
        print(f"Saving animation to {save_path} using FFMPEG chunked streaming (ZERO Memory Leak)...")
        # Bind the figure to the Agg backend canvas explicitly for offscreen rendering
        canvas = FigureCanvasAgg(self.fig)
        
        # Stream frames directly to an FFMPEG subprocess to create an MP4 on the fly
        with imageio.get_writer(save_path, fps=30) as writer:
            for i in range(frames):
                self.update(i)
                canvas.draw()
                
                # Extract RGB array natively from the Agg Buffer
                image = np.asarray(canvas.buffer_rgba())[..., :3]
                writer.append_data(image)
                
                if (i + 1) % 100 == 0:
                    print(f"Rendered {i + 1}/{frames} frames...")
                    
        print(f"Animation successfully saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints1/starling_brain_ep5000.pth")
    parser.add_argument("--save", type=str, default="murmuration_rl.gif")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--num-boids", type=int, default=250)
    parser.add_argument("--num-predators", type=int, default=4)
    parser.add_argument("--space-size", type=float, default=100.0)
    args = parser.parse_args()
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device_name = "cpu"
    print(f"Generating Simulation using {device_name}...")
    vis = RLVision3D(checkpoint_path=args.checkpoint, num_boids=args.num_boids, num_predators=args.num_predators, space_size=args.space_size, device=device_name)
    vis.animate(frames=args.frames, save_path=args.save)
