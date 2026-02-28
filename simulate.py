import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.agents.starling import StarlingBrain, FalconBrain

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
        
        if checkpoint_path is not None:
            # Load checkpoint directly to find the expected Critic dimensions
            boid_checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            stacked_frames = 3  # Must match training config
            if 'critic.0.weight' in boid_checkpoint:
                # Critic is built as Linear(global_obs_dim * stacked_frames, ...) so divide back out
                expected_global_obs_dim = boid_checkpoint['critic.0.weight'].shape[1] // stacked_frames
            else:
                expected_global_obs_dim = self.env.global_obs_dim
        else:
            expected_global_obs_dim = self.env.global_obs_dim
            stacked_frames = 3
            
        # Initialize RL Brains
        self.stacked_frames = stacked_frames
        self.boid_brain = StarlingBrain(
            obs_dim=18, 
            global_obs_dim=expected_global_obs_dim, 
            action_dim=3, 
            hidden_size=64,
            stacked_frames=self.stacked_frames
        ).to(self.device)
        
        # Determine FalconBrain's global_obs_dim from its own checkpoint
        pred_chkpt_path = checkpoint_path.replace("starling_brain", "falcon_brain") if checkpoint_path else None
        if pred_chkpt_path and os.path.exists(pred_chkpt_path):
            pred_checkpoint = torch.load(pred_chkpt_path, map_location=self.device, weights_only=True)
            if 'critic.0.weight' in pred_checkpoint:
                pred_global_obs_dim = pred_checkpoint['critic.0.weight'].shape[1] // stacked_frames
            else:
                pred_global_obs_dim = expected_global_obs_dim
        else:
            pred_checkpoint = None
            pred_global_obs_dim = expected_global_obs_dim
        
        self.pred_brain = FalconBrain(
            obs_dim=45,
            global_obs_dim=pred_global_obs_dim,
            action_dim=3,
            hidden_size=128,
            stacked_frames=self.stacked_frames
        ).to(self.device)
        
        if checkpoint_path is not None:
            self.boid_brain.load_state_dict(boid_checkpoint, strict=False)
            
            if pred_checkpoint is not None:
                self.pred_brain.load_state_dict(pred_checkpoint, strict=False)
                print(f"Loaded FalconBrain checkpoint: {pred_chkpt_path}")
            else:
                print(f"Warning: FalconBrain checkpoint not found. Predators will fly randomly.")
                
        self.boid_brain.eval()
        self.pred_brain.eval()
        
        self.obs_boids, self.obs_preds = self.env.reset()
        global_obs_boids = self.env.get_global_state(self.obs_boids)
        global_obs_preds = self.env.get_global_state(self.obs_preds)
        
        # Initialize temporal frame buffers for inference
        self.rolling_obs_boids = self.obs_boids.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        self.rolling_global_boids = global_obs_boids.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        self.rolling_obs_preds = self.obs_preds.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        self.rolling_global_preds = global_obs_preds.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        
        # Matplotlib Setup
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0, space_size])
        self.ax.set_ylim([0, space_size])
        self.ax.set_zlim([0, space_size])
        # Enable grid but keep dark theme aesthetics
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        
        # Style the grid lines and panes
        self.ax.grid(True, color='#444444', linestyle='--', alpha=0.5)
        self.ax.xaxis.pane.set_facecolor('#1e1e1e')
        self.ax.yaxis.pane.set_facecolor('#1e1e1e')
        self.ax.zaxis.pane.set_facecolor('#1e1e1e')
        self.ax.xaxis.pane.set_edgecolor('#444444')
        self.ax.yaxis.pane.set_edgecolor('#444444')
        self.ax.zaxis.pane.set_edgecolor('#444444')
        
        # Style the labels/ticks
        self.ax.tick_params(colors='#888888')
        self.ax.xaxis.label.set_color('#888888')
        self.ax.yaxis.label.set_color('#888888')
        self.ax.zaxis.label.set_color('#888888')
        
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
        # Shift temporal frame buffers left and insert new observed frame at the end
        if frame > 0:
            global_obs_boids = self.env.get_global_state(self.obs_boids)
            global_obs_preds = self.env.get_global_state(self.obs_preds)
            self.rolling_obs_boids = torch.cat([self.rolling_obs_boids[:, 1:, :], self.obs_boids.unsqueeze(1)], dim=1)
            self.rolling_global_boids = torch.cat([self.rolling_global_boids[:, 1:, :], global_obs_boids.unsqueeze(1)], dim=1)
            self.rolling_obs_preds = torch.cat([self.rolling_obs_preds[:, 1:, :], self.obs_preds.unsqueeze(1)], dim=1)
            self.rolling_global_preds = torch.cat([self.rolling_global_preds[:, 1:, :], global_obs_preds.unsqueeze(1)], dim=1)
            
        with torch.no_grad():
            N = self.env.n_agents
            P = self.env.num_predators
            
            flat_obs_boids = self.rolling_obs_boids.view(N, -1)
            features_boids = self.boid_brain.actor_feature_extractor(flat_obs_boids)
            action_batch = self.boid_brain.actor_mean(features_boids)
            
            flat_obs_preds = self.rolling_obs_preds.view(P, -1)
            features_preds = self.pred_brain.actor_feature_extractor(flat_obs_preds)
            pred_action_batch = self.pred_brain.actor_mean(features_preds)
            
        # Step Vector Env directly with Tensor
        self.obs_boids, self.obs_preds, rewards, pred_rewards, dones = self.env.step(action_batch, predator_actions=pred_action_batch)
        
        # Flush temporal history for agents that died this frame
        if dones.any():
            dead_mask = dones.bool()
            self.rolling_obs_boids[dead_mask] = self.obs_boids[dead_mask].unsqueeze(1).repeat(1, self.stacked_frames, 1)
        
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save", type=str, default="murmuration_rl.gif")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--num-boids", type=int, default=100)
    parser.add_argument("--num-predators", type=int, default=10)
    parser.add_argument("--space-size", type=float, default=50.0)
    args = parser.parse_args()
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device_name = "cpu"
    print(f"Generating Simulation using {device_name}...")
    vis = RLVision3D(checkpoint_path=args.checkpoint, num_boids=args.num_boids, num_predators=args.num_predators, space_size=args.space_size, device=device_name)
    vis.animate(frames=args.frames, save_path=args.save)
