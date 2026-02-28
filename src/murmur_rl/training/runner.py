import os
import gc
import torch
import wandb

import argparse

from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.agents.starling import StarlingBrain, FalconBrain
from murmur_rl.training.ppo import AlternatingCoevolutionTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--start-epoch", type=int, default=None, help="Epoch to start training from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    # --- 1. Hyperparameter Configuration ---
    config = {
        "num_agents": 100,           # Slightly fewer to keep FPS high during heavy training
        "num_predators": 10,
        "space_size": 50.0,
        "perception_radius": 15.0,
        "base_speed": 5.0,
        "max_turn_angle": 0.5,
        "max_force": 2.0,
        
        "rollout_steps": 500,        # Timesteps collected before PPO update
        "num_epochs": 15000,          # Total iterations
        
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,            # High entropy for exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 4,          # PPO passes over the rollout buffer
        "batch_size": 1024,          # Large batch for stable gradient
        
        "stacked_frames": 3,         # POMDP History context
    }

    # Initialize standard Weights and Biases project
    use_wandb = False
    if not args.no_wandb:
        try:
            wandb.init(
                project="murmur_rl",
                name="co-evolution run",
                config=config,
                mode="online"
            )
            use_wandb = True
        except Exception as e:
            print(f"Warning: Failed to initialize W&B ({e}). Running without W&B.")

    device_name = "cpu"
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    device = torch.device(device_name)
    print(f"Starting training on {device}...")

    # --- 2. Initialize Vectorized Environment ---
    env = VectorMurmurationEnv(
        num_agents=config["num_agents"],
        num_predators=config["num_predators"],
        space_size=config["space_size"],
        perception_radius=config["perception_radius"],
        device=device_name,
    )
    
    # Override physics engine with new biological limits from config
    env.physics.base_speed = config["base_speed"]
    env.physics.max_turn_angle = config["max_turn_angle"]
    env.physics.max_force = config["max_force"]
    
    # --- 3. Initialize Shared Brains ---
    dummy_obs_boids, dummy_obs_preds = env.reset()
    dummy_global_obs_boids = env.get_global_state(dummy_obs_boids)
    dummy_global_obs_preds = env.get_global_state(dummy_obs_preds)
    
    boid_obs_dim = dummy_obs_boids.shape[-1]
    boid_global_obs_dim = dummy_global_obs_boids.shape[-1]
    boid_brain = StarlingBrain(
        obs_dim=boid_obs_dim, 
        global_obs_dim=boid_global_obs_dim, 
        action_dim=3, 
        hidden_size=64, 
        stacked_frames=config["stacked_frames"]
    )
    
    pred_obs_dim = dummy_obs_preds.shape[-1]
    pred_global_obs_dim = dummy_global_obs_preds.shape[-1]
    pred_brain = FalconBrain(
        obs_dim=pred_obs_dim,
        global_obs_dim=pred_global_obs_dim,
        action_dim=3,
        hidden_size=128,
        stacked_frames=config["stacked_frames"]
    )
    
    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoints: {args.resume}")
        try:
            boid_chkpt = args.resume
            pred_chkpt = args.resume.replace("starling_brain", "falcon_brain")
            boid_brain.load_state_dict(torch.load(boid_chkpt, map_location=device, weights_only=True))
            if os.path.exists(pred_chkpt):
                pred_brain.load_state_dict(torch.load(pred_chkpt, map_location=device, weights_only=True))
                print(f"Loaded FalconBrain checkpoint: {pred_chkpt}")
            else:
                print(f"Warning: FalconBrain checkpoint not found at {pred_chkpt}. Starting with random predator weights.")
                
            if args.start_epoch is None:
                import re
                match = re.search(r'ep(\d+)', args.resume)
                if match:
                    start_epoch = int(match.group(1)) + 1
                    print(f"Resuming at epoch {start_epoch}")
            else:
                start_epoch = args.start_epoch
                print(f"Starting from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint {args.resume}: {e}")
            import sys
            sys.exit(1)
    
    # --- 3b. PyTorch-level optimizations ---
    # TF32 for CUDA (uses Tensor Cores for ~2x matmul throughput, negligible precision loss)
    if device_name == "cuda":
        torch.set_float32_matmul_precision("high")

    # torch.compile: fuse small ops into optimized kernels
    from murmur_rl.envs.vector_env import _HAS_TRITON
    if device_name == "cuda" and not _HAS_TRITON:
        print("  brain compile skipped (CUDA requires Triton, not installed)")
    else:
        mode = "reduce-overhead" if _HAS_TRITON else "default"
        try:
            boid_brain = torch.compile(boid_brain, mode=mode)
            pred_brain = torch.compile(pred_brain, mode=mode)
            print(f"  brains compiled (mode={mode})")
        except Exception as e:
            print(f"  brain compile skipped ({e})")
    env.compile()
    print("torch.compile setup complete")

    # --- 4. Initialize PPO Trainer ---
    trainer = AlternatingCoevolutionTrainer(
        env=env,
        boid_brain=boid_brain,
        pred_brain=pred_brain,
        device=device,
        boid_actor_lr=config["actor_lr"],
        pred_actor_lr=config["actor_lr"] / 3.0, # Slow down predator learning slightly for curriculum
        critic_lr=config["critic_lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_coef=config["clip_coef"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        update_epochs=config["update_epochs"],
        batch_size=config["batch_size"],
        stacked_frames=config["stacked_frames"],
    )
    
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # --- 5. Training Loop ---
    # Observation column indices for Boids:
    offset = (config["stacked_frames"] - 1) * boid_obs_dim
    COL_PREDATOR_DIST = 11 + offset
    COL_LOCAL_DENSITY = 4 + offset

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        
        # Collect joint experiences
        boid_rollouts, pred_rollouts = trainer.collect_rollouts(num_steps=config["rollout_steps"])
        
        # Calculate custom biological metrics from the tensor buffer
        mean_predator_dist_norm = boid_rollouts["obs"][:, :, COL_PREDATOR_DIST].mean().item()
        actual_predator_dist = mean_predator_dist_norm * (config["space_size"] / 2.0)
        
        mean_local_density_norm = boid_rollouts["obs"][:, :, COL_LOCAL_DENSITY].mean().item()
        actual_social_neighbors = mean_local_density_norm * config["num_agents"]
        
        # Linearly decay entropy coefficient over time to encourage convergence
        progress = min(1.0, (epoch - 1) / 1000.0)
        current_ent_coef = config["ent_coef"] * (1.0 - progress)
        trainer.ent_coef = current_ent_coef
        
        # Train PPO (Alternating backprop)
        metrics = trainer.train_step(boid_rollouts, pred_rollouts)
        
        # Log to WandB
        if use_wandb:
            try:
                wandb.log({
                    "epoch": epoch,
                    # Boid Metrics
                    "boid_loss/policy_loss": metrics["boids"][0],
                    "boid_loss/value_loss": metrics["boids"][1],
                    "boid_loss/entropy": metrics["boids"][2],
                    "boid_loss/explained_variance": metrics["boids"][4],
                    "boid_reward/mean_gae_return": metrics["boids"][3],
                    # Predator Metrics
                    "pred_loss/policy_loss": metrics["preds"][0],
                    "pred_loss/value_loss": metrics["preds"][1],
                    "pred_loss/entropy": metrics["preds"][2],
                    "pred_loss/explained_variance": metrics["preds"][4],
                    "pred_reward/mean_gae_return": metrics["preds"][3],
                    # Ecosystem Config
                    "biology/mean_predator_distance": actual_predator_dist,
                    "biology/mean_social_neighbors": actual_social_neighbors,
                })
            except Exception as e:
                print(f"Warning: WandB log failed at epoch {epoch} ({e})")
        
        if epoch % 50 == 0 or epoch == start_epoch:
            b_ret, b_ent, b_vloss, b_ploss, _ = metrics["boids"]
            p_ret, p_ent, p_vloss, p_ploss, _ = metrics["preds"]
            print(f"Epoch {epoch:04d} | Cohort: {actual_social_neighbors:>4.1f} | EvasionDist: {actual_predator_dist:>5.1f}m")
            print(f"  [BOIDS] Ret: {b_ret:>7.4f} | Ent: {b_ent:>6.4f} | VLoss: {b_vloss:>7.4f} | Ploss: {b_ploss:>7.4f}")
            print(f"  [PREDS] Ret: {p_ret:>7.4f} | Ent: {p_ent:>6.4f} | VLoss: {p_vloss:>7.4f} | Ploss: {p_ploss:>7.4f}")
        
        # Checkpointing
        if epoch % 500 == 0:
            boid_chkpt_path = f"{args.checkpoints_dir}/starling_brain_ep{epoch}.pth"
            pred_chkpt_path = f"{args.checkpoints_dir}/falcon_brain_ep{epoch}.pth"
            torch.save(boid_brain.state_dict(), boid_chkpt_path)
            torch.save(pred_brain.state_dict(), pred_chkpt_path)
            print(f"Saved Checkpoints: {boid_chkpt_path} & {pred_chkpt_path}")
            
        # Explicit Memory Management for Apple Silicon
        # MPS aggressively caches tensor allocations, which looks like a giant RAM leak
        # over thousands of epochs if not manually cleared.
        del boid_rollouts, pred_rollouts
        gc.collect()
        if device_name == 'mps':
            torch.mps.empty_cache()
            
    if use_wandb:
        try:
            wandb.finish()
        except:
            pass
            
    print("Training Complete. Final model saved.")
    
if __name__ == "__main__":
    main()
