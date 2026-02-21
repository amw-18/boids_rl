import torch
from murmur_rl.envs.murmuration import MurmurationEnv
from murmur_rl.agents.starling import StarlingBrain
from murmur_rl.training.ppo import PPOTrainer

def main():
    device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"Starting training on {device}...")

    # 1. Initialize Vectorized Environment
    env = MurmurationEnv(
        num_agents=100,
        space_size=100.0,
        perception_radius=15.0,
        device=device_name
    )
    
    # 2. Initialize Brain
    # obs_dim = 16 (Biological Group Context + Perceptual Threat)
    obs_dim = 16
    brain = StarlingBrain(obs_dim=obs_dim, action_dim=3, hidden_size=64)
    
    # 3. Initialize PPO Trainer
    trainer = PPOTrainer(
        env=env,
        brain=brain,
        device=device,
        lr=3e-4,
        batch_size=128,
        update_epochs=4
    )
    
    # 4. Training Loop
    num_iterations = 10
    rollout_steps = 100
    
    for i in range(num_iterations):
        print(f"--- Iteration {i+1}/{num_iterations} ---")
        
        # Collect experiences from all agents concurrently
        rollouts = trainer.collect_rollouts(num_steps=rollout_steps)
        print(f"Collected {rollouts['obs'].shape[0]} agent-steps of data.")
        
        # Train
        pg_loss, v_loss, mean_return = trainer.train_step(rollouts)
        print(f"Policy Loss: {pg_loss:.4f} | Value Loss: {v_loss:.4f} | Mean Return (GAE): {mean_return:.4f}")
        
    print("Training pipeline test completed successfully.")
    
if __name__ == "__main__":
    main()
