import numpy as np
import torch
import mujoco
from ball_and_stick_rl.sac import SphericalPendulumEnv, CustomSAC, PolicyNetwork
import mujoco.viewer
import time

# Create the environment
env = SphericalPendulumEnv(randomize_velocity=False, max_speed=0.5)

# Initialize the policy network
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = PolicyNetwork(obs_dim, act_dim, hidden_size=32, num_layers=1)

# Load the trained model
checkpoint_path = "checkpoints/model_sac.pt"  # Path to SAC checkpoint
checkpoint = torch.load(checkpoint_path)
policy.load_state_dict(checkpoint["policy"])  # Load only the policy network
policy.eval()  # Set to evaluation mode
policy.to("cpu")  # Adjust to "cuda" if GPU is available

# Create CustomSAC instance
model = CustomSAC(
    env,
    learning_rate=1e-4,
    buffer_size=1000000,
    batch_size=128,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    hidden_size=32,
    num_layers=1,
    device="cpu",
)
model.policy = policy  # Assign the loaded policy to the model

# Launch passive viewer
viewer = mujoco.viewer.launch_passive(env.model, env.data)

# Get simulation parameters
timestep = env.model.opt.timestep  # MuJoCo timestep (0.001)
frame_skip = env.frame_skip  # Number of simulation steps per control step
control_period = timestep * frame_skip  # Effective control period in seconds

try:
    while True:
        obs, _ = env.reset()
        hidden = model.policy.init_hidden().to(model.device)
        target_velocity = np.random.uniform(low=-0.3, high=0.3, size=2)

        # Forward simulation to update viewer
        mujoco.mj_forward(env.model, env.data)
        viewer.sync()

        # Initialize timing
        start_time = time.time()

        for t in range(10000):
            # Get current time
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Predict with custom SAC policy
            action, hidden = model.predict(obs, hidden=hidden, deterministic=True)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(
                action=action, target_velocity=target_velocity
            )

            print(
                f"Step {t}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Target Velocity: {target_velocity}"
            )

            # Sync viewer with current data
            viewer.sync()

            # Synchronize with real time
            elapsed_time = time.time() - start_time
            target_time = (t + 1) * control_period  # Expected time for the next step
            sleep_time = target_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)  # Wait to match the desired control period

            # Check if episode is done
            if terminated or truncated:
                break

finally:
    # Clean up
    viewer.close()
    env.close()
