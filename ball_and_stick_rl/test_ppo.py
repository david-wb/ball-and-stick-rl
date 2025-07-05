import numpy as np
from sb3_contrib import RecurrentPPO
from ball_and_stick_rl.recurrent import SphericalPendulumEnv  # Adjust if needed
import mujoco.viewer

# Load the trained recurrent model
model = RecurrentPPO.load("checkpoints/model.zip")
model.policy.to("cuda")

# Create the environment
env = SphericalPendulumEnv()

# Launch passive viewer
viewer = mujoco.viewer.launch_passive(env.model, env.data)

episode_start = np.ones((1,), dtype=bool)

while True:
    obs, _ = env.reset()
    mujoco.mj_forward(env.model, env.data)
    viewer.sync()
    state = None
    for t in range(10000):
        if t == 0:
            target_velocity = np.random.uniform(low=-0.5, high=0.5, size=2)
            env.target_velocity = target_velocity

        # Predict with recurrent policy
        action, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action, target_velocity=target_velocity
        )

        print(
            f"Step {t}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, target_velocity: {target_velocity}"
        )

        # Sync viewer with current data
        viewer.sync()

        # Update episode_start flag
        episode_done = terminated or truncated
        episode_start = np.array([episode_done], dtype=bool)

        if episode_done:
            break

# Clean up
viewer.close()
env.close()
