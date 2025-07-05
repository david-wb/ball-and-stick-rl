import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
import wandb


MUJOCO_XML = """
<mujoco model="spherical_pendulum">
    <compiler angle="radian" coordinate="local"/>
    <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81"/>
    <default>
        <geom rgba="1 1 1 1" friction="0.8 0.05 0.001"/>
        <joint damping="0.2"/>
    </default>
    <visual>
        <global offwidth="1920" offheight="1080"/>
    </visual>
    <asset>
        <texture name="checker" type="2d" builtin="checker" rgb1="0 0 0" rgb2="1 1 1" width="100" height="100" mark="none"/>
        <material name="checker_mat" texture="checker" specular="0.5" shininess="0.6" texrepeat="10 10"/>
    </asset>
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1" specular="0 0 0"/>
        <geom name="floor" type="plane" material="checker_mat" size="10 10 0.1"/>
        <body name="base" pos="0 0 0.15">
            <freejoint name="base_free"/>
            <geom name="base_sphere" type="sphere" size="0.15" mass="10" rgba="0.3 0.3 0.3 1"/>
            <body name="pendulum" pos="0 0 0">
                <joint name="ball_joint" type="ball" pos="0 0 0" range="0 3.1416" damping="0.2"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 1.0" size="0.01" mass="1.5" rgba="0 0.7 0.7 1"/>
                <geom name="tip_sphere" type="sphere" pos="0 0 1.0" size="0.05" mass="0" rgba="0.7 0.3 0.3 1"/>
                <site name="tip" pos="0 0 1.0" size="0.02"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="torque_x" joint="ball_joint" gear="1 0 0" ctrlrange="-50 50"/>
        <motor name="torque_y" joint="ball_joint" gear="0 1 0" ctrlrange="-50 50"/>
    </actuator>
    <sensor>
        <framequat name="angle" objtype="body" objname="pendulum"/>
        <frameangvel name="velocity" objtype="body" objname="pendulum"/>
        <framepos name="base_pos" objtype="body" objname="base"/>
        <framexaxis name="base_xaxis" objtype="body" objname="base"/>
        <frameyaxis name="base_yaxis" objtype="body" objname="base"/>
        <framezaxis name="base_zaxis" objtype="body" objname="base"/>
    </sensor>
</mujoco>
"""

xml_file = "spherical_pendulum.xml"
if not os.path.isfile(xml_file):
    with open(xml_file, "w") as f:
        f.write(MUJOCO_XML)


class SphericalPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self, render_mode=None, max_speed: float = 0.5, randomize_velocity=True
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_speed = max_speed
        self.randomize_velocity = randomize_velocity
        self.target_velocity = np.array([0.0, 0.0], dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low=-50.0, high=50.0, shape=(2,), dtype=np.float32
        )
        obs_size = 3 + 3 + 3 + 2  # z_axis, ang_vel, base_vel, target_velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.viewer = None
        self.frame_skip = 4

    def _get_obs(self):
        quat = self.data.sensor("angle").data
        w, x, y, z = quat
        z_axis = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        ang_vel = self.data.sensor("velocity").data
        base_vel = self.data.qvel[:3]
        obs = np.concatenate(
            [
                z_axis / np.linalg.norm(z_axis),
                ang_vel / 1.0,  # Assume max angular velocity ~1.0
                base_vel / self.max_speed,  # Max base velocity
                self.target_velocity / self.max_speed,
            ]
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.15])

        quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat[0] += np.random.uniform(-0.1, 0.1)
        quat /= np.linalg.norm(quat)
        self.data.qpos[3:7] = quat

        if self.randomize_velocity:
            speed = np.random.uniform(0, self.max_speed)
            angle = np.random.uniform(0, 2 * np.pi)
            self.target_velocity = speed * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
        else:
            self.target_velocity = np.array([0.0, 0.0], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action, target_velocity=None):
        if target_velocity is not None:
            self.target_velocity = np.array(target_velocity, dtype=np.float32)

        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        z_axis = obs[:3]  # Normalized z_axis from _get_obs
        upright_reward = z_axis[2]  # Cosine of angle with vertical
        angle_deviation = np.arccos(np.clip(z_axis[2], -1, 1))
        angle_penalty = -1.0 * angle_deviation / (np.pi/2)  # Scales between [-1, 0]
        base_vel = obs[6:9][:2]  # Normalized base velocity (x, y components)
        vel_error = np.linalg.norm(
            base_vel * self.max_speed - self.target_velocity
        )  # Un-normalize base_vel
        max_vel_error = self.max_speed  # From velocity_range
        velocity_reward = -1.0 * (vel_error / max_vel_error)
        control_penalty = -0.01 * np.sum(np.square(action))

        reward = float(10 * angle_penalty + velocity_reward + control_penalty)
        reward = np.clip(reward, -10, 10)  # Clip reward to stabilize learning
        terminated = bool(z_axis[2] < 0.1)  # Relaxed termination condition
        truncated = False

        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model)
            self.viewer.update_scene(self.data)
            self.viewer.render()

        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                "upright_reward": upright_reward,
                "angle_penalty": angle_penalty,
                "velocity_reward": velocity_reward,
                "control_penalty": control_penalty,
            },
        )

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model)
            self.viewer.update_scene(self.data)
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"\nSaved model checkpoint to: {save_file}")
        return True


if __name__ == "__main__":
    env = SphericalPendulumEnv(randomize_velocity=True)
    env = Monitor(env)
    check_env(env)
    vec_env = DummyVecEnv([lambda: env])

    model = RecurrentPPO(
        "MlpLstmPolicy",  # <- GRU Policy
        vec_env,
        device="cuda",
        verbose=1,
        n_steps=8192,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        learning_rate=1e-4,
        tensorboard_log="./tensorboard/",
    )

    save_callback = SaveOnStepCallback(save_freq=100_000, save_path="./checkpoints")

    run = wandb.init(
        project="spherical-pendulum",
        config={
            "algo": "RecurrentPPO",
            "env": "SphericalPendulumEnv",
            "total_timesteps": 10_000_000,
            "learning_rate": model.learning_rate,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "clip_range": model.clip_range,
            "ent_coef": model.ent_coef,
        },
        sync_tensorboard=True,
        monitor_gym=True,
    )

    callbacks = CallbackList(
        [
            save_callback,
            WandbCallback(model_save_path="./wandb_models", verbose=1),
        ]
    )

    model.learn(total_timesteps=10_000_000, progress_bar=True, callback=callbacks)
    model.save("spherical_pendulum_recurrentppo_final")
    wandb.finish()

    # --- Evaluation ---
    eval_env = SphericalPendulumEnv(render_mode="human", randomize_velocity=False)
    obs, _ = eval_env.reset()
    state = model.policy.initial_state(batch_size=1)
    episode_start = np.ones((1,), dtype=bool)

    for t in range(1000):
        angle = 2 * np.pi * (t % 100) / 100
        target_velocity = 0.3 * np.array([np.cos(angle), np.sin(angle)])

        action, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=True
        )
        obs, reward, terminated, truncated, info = eval_env.step(
            action, target_velocity=target_velocity
        )

        episode_start = np.array([terminated or truncated], dtype=bool)
        eval_env.render()

        if terminated or truncated:
            obs, _ = eval_env.reset()
            state = model.policy.initial_state(batch_size=1)
            episode_start = np.ones((1,), dtype=bool)

    eval_env.close()
