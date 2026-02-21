import torch
import time
from murmur_rl.envs.physics import BoidsPhysics

def test_initialization():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_boids = 1000
    env = BoidsPhysics(num_boids=num_boids, device=device)
    
    assert env.positions.shape == (num_boids, 3)
    assert env.velocities.shape == (num_boids, 3)
    print("Testing initialization: PASSED")

def test_step_performance():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    num_boids = 1024  # Good number for testing parallel performance
    steps = 100
    
    env = BoidsPhysics(num_boids=num_boids, device=device)
    
    # Warmup
    env.step()
    
    if device.type == 'mps':
        torch.mps.synchronize()
        
    start_time = time.time()
    for _ in range(steps):
        env.step()
        
    if device.type == 'mps':
        torch.mps.synchronize()
        
    end_time = time.time()
    
    fps = steps / (end_time - start_time)
    print(f"Tested 3D simulation with {num_boids} boids for {steps} steps on {device}.")
    print(f"Performance: {fps:.2f} FPS (Steps per second)")

if __name__ == "__main__":
    test_initialization()
    test_step_performance()
