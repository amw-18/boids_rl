import os
import gc
import torch
import numpy as np
import wandb

from murmur_rl.envs.murmuration import MurmurationEnv
from murmur_rl.agents.starling import StarlingBrain
from murmur_rl.training.ppo import PPOTrainer
from murmur_rl.utils.render import BoidsVisualizer3D

def main():
    # --- 1. Hyperparameter Configuration ---
    config = {
        "num_agents": 250,           # Slightly fewer to keep FPS high during heavy training
        "space_size": 100.0,
        "perception_radius": 15.0,
        "base_speed": 5.0,
        "max_turn_angle": 0.5,
        "max_force": 2.0,
        
        "rollout_steps": 100,        # Timesteps collected before PPO update
        "num_epochs": 5000,          # Total iterations
        
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,            # High entropy for exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 4,          # PPO passes over the rollout buffer
        "batch_size": 1024,          # Large batch for stable gradient
    }

    # Initialize standard Weights and Biases project
    wandb.init(
        project="murmur_rl",
        name="ppo_continuous_run_cuda",
        config=config,
        mode="online"  # Disabled so it doesn't crash on unauthenticated local machines
    )

    device_name = "cpu"
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    device = torch.device(device_name)
    print(f"Starting training on {device}...")

    # --- 2. Initialize Vectorized Environment ---
    env = MurmurationEnv(
        num_agents=config["num_agents"],
        space_size=config["space_size"],
        perception_radius=config["perception_radius"],
        device=device_name
    )
    
    # Override physics engine with new biological limits from config
    env.physics.base_speed = config["base_speed"]
    env.physics.max_turn_angle = config["max_turn_angle"]
    env.physics.max_force = config["max_force"]
    
    # --- 3. Initialize Shared Brain ---
    obs_dim = 16
    brain = StarlingBrain(obs_dim=obs_dim, action_dim=3, hidden_size=64)
    
    # --- 4. Initialize PPO Trainer ---
    trainer = PPOTrainer(
        env=env,
        brain=brain,
        device=device,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_coef=config["clip_coef"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        update_epochs=config["update_epochs"],
        batch_size=config["batch_size"]
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # --- 5. Training Loop ---
    for epoch in range(1, config["num_epochs"] + 1):
        
        # Collect experiences
        rollouts = trainer.collect_rollouts(num_steps=config["rollout_steps"])
        
        # Calculate custom biological metrics
        # Mean distance to predator at the end of rollout
        # Threat dist is input dim 5, normalized by (space_size/2)
        mean_predator_dist_norm = rollouts["obs"][:, 5].mean().item()
        actual_predator_dist = mean_predator_dist_norm * (config["space_size"] / 2.0)
        
        # Mean local density is input dim 2, normalized by n_agents
        mean_local_density_norm = rollouts["obs"][:, 2].mean().item()
        actual_social_neighbors = mean_local_density_norm * config["num_agents"]
        
        # Train PPO
        pg_loss, v_loss, entropy, mean_return = trainer.train_step(rollouts)
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "loss/policy_loss": pg_loss,
            "loss/value_loss": v_loss,
            "loss/entropy": entropy,
            "reward/mean_gae_return": mean_return,
            "biology/mean_predator_distance": actual_predator_dist,
            "biology/mean_social_neighbors": actual_social_neighbors,
        })
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Ret: {mean_return:>7.4f} | Ent: {entropy:>6.4f} | VLoss: {v_loss:>7.4f} | Ploss: {pg_loss:>7.4f} | EvasionDist: {actual_predator_dist:>5.1f}m | Cohort: {actual_social_neighbors:>4.1f}")
        
        # Checkpointing
        if epoch % 500 == 0:
            chkpt_path = f"checkpoints/starling_brain_ep{epoch}.pth"
            torch.save(brain.state_dict(), chkpt_path)
            print(f"Saved Checkpoint: {chkpt_path}")
            
        # Explicit Memory Management for Apple Silicon
        # MPS aggressively caches tensor allocations, which looks like a giant RAM leak
        # over thousands of epochs if not manually cleared.
        del rollouts
        gc.collect()
        if device_name == 'mps':
            torch.mps.empty_cache()
            
    wandb.finish()
    print("Training Complete. Final model saved.")
    
if __name__ == "__main__":
    main()
