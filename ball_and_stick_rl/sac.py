import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
from torch.distributions.normal import Normal
import uuid
from collections import deque
import random

# MuJoCo XML model string
MUJOCO_XML = """
<mujoco model="robot_with_wheelbase">
    <compiler coordinate="local" angle="degree"/>
    <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81" impratio="10"/>
    <default>
        <geom rgba="1 1 1 1" />
        <joint damping="0.2"/>
    </default>
    <visual>
        <global offwidth="1920" offheight="1080"/>
    </visual>
    <asset>
        <texture name="checker" type="2d" builtin="checker" rgb1="0 0 0" rgb2="1 1 1" width="100" height="100" mark="none"/>
        <material name="checker_mat" texture="checker" specular="0.5" shininess="0.6" texrepeat="10 10"/>
        <material name="robot_mat" rgba="0.5 0.5 0.5 1"/>
    </asset>
    <worldbody>
        <geom name="floor" type="plane" material="checker_mat" size="10 10 0.1"/>
        <body name="base" pos="0 0 0.15">
            <freejoint name="base_free"/>
            <geom name="base_sphere" type="sphere" size="0.15" mass="20" rgba="0.4 0.5 0.6 0.5" />
        </body>
        <body name="wheel_base" pos="0 0 0.32">
            <freejoint name="wheel_base_free"/>
            <!-- Robot Chassis -->
            <geom name="chassis_geom" type="cylinder" size="0.15 0.01" material="robot_mat" mass="1.0" rgba="1 0.5 0 0.5"/>
            <!-- Vertical pendulum -->
            <body name="pendulum" pos="0 0 0.3">
                <geom name="rod_geom" type="cylinder" size="0.01 0.3" rgba="1 0.2 0 0.5" mass="0.2"/>
                <!-- Ball Tip -->
                <body name="rod_tip" pos="0 0 0.3">
                    <geom name="rod_tip_geom" type="sphere" size="0.04" rgba="1 0.0 0 0.5" mass="0.05"/>
                </body>
            </body>
            <!-- Wheel 1: Positioned at 0 degrees, tilted -45 degrees toward ball center -->
            <body name="wheel1" pos="0.12727922061357855 0 -0.07" euler="0 -45 0">
                <joint name="wheel1_joint" type="hinge" axis="0 0 1" damping="0.1"/>
                <geom name="wheel1_geom" type="cylinder" size="0.03 0.02" rgba="0.5 0.5 0.5 1" mass="0.5" />
            </body>
            <body name="wheel2" pos="-0.06363961030678925 0.110227038425243 -0.07" euler="40.89339465 20.70481105 -112.2076543">
                <joint name="wheel2_joint" type="hinge" axis="0 0 1" damping="0.1"/>
                <geom name="wheel2_geom" type="cylinder" size="0.03 0.02" rgba="0.5 0.5 0.5 1" mass="0.5" />
            </body>
            <body name="wheel3" pos="-0.06363961030678925 -0.11022703842524297 -0.07" euler="-40.89339465 20.70481105 112.2076543">
                <joint name="wheel3_joint" type="hinge" axis="0 0 1" damping="0.1"/>
                <geom name="wheel3_geom" type="cylinder" size="0.03 0.02" rgba="0.5 0.5 0.5 1" mass="0.5" />
            </body>
        </body>
        <light name="diffuse_light" pos="0 0 5" dir="0 0 -1" directional="false" diffuse="0.8 0.8 0.8" specular="0 0 0" castshadow="false" />
    </worldbody>
    <sensor>
        <framequat name="pendulum_angle" objtype="body" objname="pendulum"/>
        <frameangvel name="pendulum_angular_velocity" objtype="body" objname="pendulum"/>
        <framelinvel name="base_linear_velocity" objtype="body" objname="base"/>
    </sensor>
    <actuator>
        <motor name="motor1" joint="wheel1_joint" ctrlrange="-5 5"/>
        <motor name="motor2" joint="wheel2_joint" ctrlrange="-5 5"/>
        <motor name="motor3" joint="wheel3_joint" ctrlrange="-5 5"/>
    </actuator>
    <contact>
        <!-- Anisotropic friction: zero along wheel z-axis, non-zero for theta -->
        <pair geom2="wheel1_geom" geom1="base_sphere" friction="0 1 0.005 0.0000 0.0000" condim="6" />
        <pair geom2="wheel2_geom" geom1="base_sphere" friction="0 1 0.005 0.0000 0.0000" condim="6" />
        <pair geom2="wheel3_geom" geom1="base_sphere" friction="0 1 0.005 0.0000 0.0000" condim="6" />
    </contact>
</mujoco>
"""

xml_file = "mujoco_models/robot.xml"

if not os.path.isfile(xml_file):
    with open(xml_file, "w") as f:
        f.write(MUJOCO_XML)


class SphericalPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        max_steps: int = 1024,
        max_speed: float = 0.5,
        randomize_velocity=True,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.max_speed = max_speed
        self.randomize_velocity = randomize_velocity
        self.target_velocity = np.array([0.0, 0.0], dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Updated action space to 3 dimensions for 3 motors
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(3,), dtype=np.float32
        )

        obs_size = 3 + 3 + 3 + 2  # z_axis, ang_vel, base_vel, target_velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.viewer = None
        self.frame_skip = 4
        self.step_count = 0

    def _get_obs(self):
        # Read pendulum quaternion to compute z-axis projection
        quat = self.data.sensor("pendulum_angle").data
        w, x, y, z = quat
        z_axis = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        # Read pendulum angular velocity
        pend_ang_vel = self.data.sensor("pendulum_angular_velocity").data
        # Read base linear velocity
        base_lin_vel = self.data.sensor("base_linear_velocity").data
        # Concatenate observation vector
        obs = np.concatenate(
            [
                z_axis / np.linalg.norm(z_axis),  # Normalized pendulum z-axis
                pend_ang_vel / 1.0,  # Normalized pendulum angular velocity
                base_lin_vel / self.max_speed,  # Normalized base linear velocity
                self.target_velocity / self.max_speed,  # Normalized target velocity
            ]
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Fully reset MuJoCo simulation state
        mujoco.mj_resetData(self.model, self.data)

        # Set initial position of the base (ball)
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.15])  # x, y, z position

        # Set initial quaternion for the base with a small random perturbation
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat[0] += np.random.uniform(-0.1, 0.1)
        quat /= np.linalg.norm(quat)
        self.data.qpos[3:7] = quat

        # Explicitly set all velocities to zero
        self.data.qvel[:] = 0.0

        # Explicitly set all control inputs (motor torques) to zero
        self.data.ctrl[:] = 0.0

        # Set target velocity
        if self.randomize_velocity:
            speed = np.random.uniform(0, self.max_speed)
            angle = np.random.uniform(0, 2 * np.pi)
            self.target_velocity = speed * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
        else:
            self.target_velocity = np.array([0.0, 0.0], dtype=np.float32)

        # Update the simulation to ensure a consistent state
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def angle_between_vectors(self, v1, v2):
        """
        Compute the angle in radians between two vectors using NumPy, with output in [0, pi].

        Args:
            v1 (np.ndarray): First vector
            v2 (np.ndarray): Second vector

        Returns:
            float: Angle in radians, in the range [0, pi]

        Raises:
            ValueError: If either vector has zero magnitude
        """
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            raise ValueError("One or both vectors have zero magnitude")
        cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
        return np.arccos(cos_theta)

    def step(self, action, target_velocity=None):
        self.step_count += 1
        if target_velocity is not None:
            self.target_velocity = np.array(target_velocity, dtype=np.float32)

        # Ensure action is 3D and assign to all three motors
        action = np.clip(action, -5.0, 5.0)  # Ensure action is within bounds
        self.data.ctrl[:3] = action  # Assign to motor1, motor2, motor3
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        z_axis = obs[:3]  # Normalized z_axis from _get_obs
        angle_deviation = (
            np.arccos(np.clip(z_axis[2], -1, 1)) / np.pi
        )  # Normalize angle deviation to [0, 1]

        upright_reward = 1 - angle_deviation

        base_vel = obs[6:9][:2]  # Normalized base velocity (x, y components)

        vel_angle_error = (
            self.angle_between_vectors(base_vel, self.target_velocity) / np.pi
        )  # Normalize angle error to [0, 1]

        vel_speed_error = (
            abs(
                np.linalg.norm(base_vel * self.max_speed)
                - np.linalg.norm(self.target_velocity)
            )
            / self.max_speed
        )

        if angle_deviation < np.pi / 6:
            vel_angle_reward = 0.2 * (1 - vel_angle_error)
            vel_speed_reward = 0.1 * (1 - np.clip(vel_speed_error, 0, 1))
        else:
            vel_angle_reward = 0
            vel_speed_reward = 0

        # Update control penalty for 3D action
        control_penalty = -0.001 * np.sum(np.square(action))

        reward = float(
            2 * upright_reward + vel_angle_reward + vel_speed_reward + control_penalty
        )
        terminated = bool(z_axis[2] < 0.1)  # Relaxed termination condition

        # Check if robot fell or jumped off the sphere
        wheel_base_pos = self.data.body("wheel_base").xpos
        if wheel_base_pos[2] < 0.2:
            terminated = True
            reward = -1.0

        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

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
                "vel_angle_reward": vel_angle_reward,
                "vel_speed_reward": vel_speed_reward,
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


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, hidden_dim, seq_len, device):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Store sequences
        self.observations = np.zeros((capacity, seq_len, obs_dim), dtype=np.float32)
        self.hiddens = np.zeros((capacity, seq_len, 1, hidden_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, seq_len, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, seq_len), dtype=np.float32)
        self.next_observations = np.zeros(
            (capacity, seq_len, obs_dim), dtype=np.float32
        )
        self.next_hiddens = np.zeros(
            (capacity, seq_len, 1, hidden_dim), dtype=np.float32
        )
        self.dones = np.zeros((capacity, seq_len), dtype=np.bool_)
        self.masks = np.ones(
            (capacity, seq_len), dtype=np.float32
        )  # Mask for valid transitions
        self.index = 0
        self.seq_index = 0
        self.size = 0

    def push(self, obs, hidden, act, rew, next_obs, next_hidden, done, truncate):
        if self.seq_index < self.seq_len:
            if self.seq_index == 0:
                # Initialize the sequence at the start of a new episode
                self.observations[self.index] = 0
                self.hiddens[self.index] = 0
                self.actions[self.index] = 0
                self.rewards[self.index] = 0
                self.next_observations[self.index] = 0
                self.next_hiddens[self.index] = 0
                self.dones[self.index] = True
                self.masks[self.index] = 0

            self.observations[self.index][self.seq_index] = obs
            self.hiddens[self.index][self.seq_index] = hidden
            self.actions[self.index][self.seq_index] = act
            self.rewards[self.index][self.seq_index] = rew
            self.next_observations[self.index][self.seq_index] = next_obs
            self.next_hiddens[self.index][self.seq_index] = next_hidden
            self.dones[self.index][self.seq_index] = done
            self.masks[self.index][self.seq_index] = 1.0

        if self.seq_index + 1 >= self.seq_len or done or truncate:
            self.index = (self.index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.seq_index = 0
        else:
            self.seq_index += 1

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.observations[indices]).to(self.device),
            torch.FloatTensor(self.hiddens[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_observations[indices]).to(self.device),
            torch.FloatTensor(self.next_hiddens[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
            torch.FloatTensor(self.masks[indices]).to(self.device),
        )


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=32, num_layers=1):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer for processing sequential observations
        self.gru = nn.GRU(obs_dim, hidden_size, num_layers, batch_first=True)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )
        self.actor_log_std = nn.Parameter(-torch.ones(act_dim))  # Learnable log_std
        self.action_scale = torch.tensor(5.0)  # Scale to [-5, 5]

    def forward(self, obs, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, obs_dim)
        if hidden is None:
            hidden = self.init_hidden(obs.size(0)).to(obs.device)
        gru_out, new_hidden = self.gru(obs, hidden)
        mean = self.actor_mean(gru_out) * self.action_scale.to(gru_out.device)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std, new_hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=32):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs, act):
        reshape = False
        if obs.dim() == 3:  # (batch_size, seq_len, obs_dim)
            reshape = True
            batch_size, seq_len, _ = obs.size()
            obs = obs.view(-1, obs.size(-1))  # (batch_size * seq_len, obs_dim)
            act = act.view(-1, act.size(-1))  # (batch_size * seq_len, act_dim)
        x = torch.cat([obs, act], dim=-1)
        q_values = self.network(x)
        if reshape:
            q_values = q_values.view(batch_size, seq_len, 1)  # (batch_size, seq_len, 1)
        return q_values


class CustomSAC:
    def __init__(
        self,
        env: SphericalPendulumEnv,
        learning_rate=1e-4,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.5,
        hidden_size=32,
        num_layers=1,
        seq_len=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]  # Now 3 for three motors
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Initialize networks
        self.policy = PolicyNetwork(
            self.obs_dim, self.act_dim, hidden_size, num_layers
        ).to(self.device)
        self.q1 = QNetwork(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.target_q1 = QNetwork(self.obs_dim, self.act_dim, hidden_size).to(
            self.device
        )
        self.target_q2 = QNetwork(self.obs_dim, self.act_dim, hidden_size).to(
            self.device
        )

        # Initialize target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.obs_dim,
            self.act_dim,
            self.hidden_size,
            seq_len,
            device=self.device,
        )

    def update_target_networks(self):
        for target_param, param in zip(
            self.target_q1.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for target_param, param in zip(
            self.target_q2.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def learn(self, total_timesteps, save_freq=10000, save_path="./checkpoints"):
        os.makedirs(save_path, exist_ok=True)
        obs, _ = self.env.reset()
        episode_rewards = []
        timestep = 0
        episode_count = 0
        hidden = self.policy.init_hidden().to(self.device)

        while timestep < total_timesteps:
            # Collect experience
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                mean, std, next_hidden = self.policy(obs_tensor, hidden)
                dist = Normal(mean, std)
                action = dist.sample()
                action_np = action.cpu().detach().numpy()[0]
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated

                # Store transition
                self.replay_buffer.push(
                    obs,
                    hidden.cpu().detach().numpy()[:, 0, :],
                    action_np,
                    reward,
                    next_obs,
                    next_hidden.cpu().detach().numpy()[:, 0, :],
                    done,
                    truncated,
                )

            # Log metrics
            wandb.log(
                {
                    "action_mean": np.mean(action_np),
                    "action_std": np.std(action_np),
                    "obs_mean": np.mean(obs),
                    "obs_std": np.std(obs),
                    "timestep": timestep,
                    **info,
                }
            )

            episode_rewards.append(reward)
            obs = next_obs
            hidden = next_hidden.detach()
            timestep += 1

            self._update_networks()

            # Reset episode if done
            if done or truncated:
                episode_count += 1
                wandb.log(
                    {
                        "episode_reward": sum(episode_rewards),
                        "episode_length": self.env.step_count,
                        "episode_count": episode_count,
                    }
                )
                episode_rewards = []
                obs, _ = self.env.reset()
                hidden = self.policy.init_hidden().to(self.device)

            # Save model checkpoint
            if timestep % save_freq == 0:
                save_file = os.path.join(save_path, f"model_sac.pt")
                torch.save(
                    {
                        "policy": self.policy.state_dict(),
                        "q1": self.q1.state_dict(),
                        "q2": self.q2.state_dict(),
                        "target_q1": self.target_q1.state_dict(),
                        "target_q2": self.target_q2.state_dict(),
                    },
                    save_file,
                )
                print(f"Saved model checkpoint to: {save_file}")

    def _update_networks(self):
        if self.replay_buffer.size < self.batch_size * 10:
            return

        (
            batch_obs,
            batch_hidden,
            batch_act,
            batch_rew,
            batch_next_obs,
            batch_next_hidden,
            batch_done,
            batch_mask,
        ) = self.replay_buffer.sample(self.batch_size)

        batch_hidden = batch_hidden[:, 0].permute(1, 0, 2)
        batch_next_hidden = batch_next_hidden[:, 0].permute(1, 0, 2)

        # Q-network updates
        with torch.no_grad():
            next_mean, next_std, _ = self.policy(batch_next_obs, batch_next_hidden)
            next_dist = Normal(next_mean, next_std)
            next_action = next_dist.sample()
            next_log_prob = next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            target_q1 = self.target_q1(batch_next_obs, next_action)
            target_q2 = self.target_q2(batch_next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            # Compute target Q-values with uniform discounting
            not_dones = (1 - batch_done.unsqueeze(-1)) * batch_mask.unsqueeze(-1)
            rewards = batch_rew.unsqueeze(-1) * batch_mask.unsqueeze(-1)
            target = rewards + not_dones * self.gamma * target_q

        q1_pred = self.q1(batch_obs, batch_act)
        q2_pred = self.q2(batch_obs, batch_act)
        q1_loss = ((q1_pred - target) ** 2 * batch_mask.unsqueeze(-1)).mean()
        q2_loss = ((q2_pred - target) ** 2 * batch_mask.unsqueeze(-1)).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy update
        mean, std, _ = self.policy(batch_obs, batch_hidden)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        q1_pi = self.q1(batch_obs, action)
        q2_pi = self.q2(batch_obs, action)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_prob - q_pi) * batch_mask.unsqueeze(-1)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target networks
        self.update_target_networks()

        # Log losses
        wandb.log(
            {
                "q1_loss": q1_loss.item(),
                "q2_loss": q2_loss.item(),
                "policy_loss": policy_loss.item(),
                "alpha": self.alpha,
            }
        )

    def predict(self, obs, hidden=None, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, new_hidden = self.policy(obs_tensor, hidden)
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
        return action.cpu().numpy()[0], new_hidden


if __name__ == "__main__":
    # Training
    env = SphericalPendulumEnv(max_steps=2048, randomize_velocity=True)
    model = CustomSAC(
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        alpha=0.3,
        hidden_size=32,
        num_layers=1,
        seq_len=32,
        device="cpu",
    )
    run = wandb.init(
        project="spherical-pendulum",
        config={
            "algo": "CustomSAC-GRU",
            "env": "SphericalPendulumEnv",
            "total_timesteps": 100_000_000,
            "learning_rate": model.learning_rate,
            "buffer_size": model.buffer_size,
            "batch_size": model.batch_size,
            "gamma": model.gamma,
            "tau": model.tau,
            "alpha": model.alpha,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "seq_len": model.seq_len,
        },
    )
    model.learn(total_timesteps=100_000_000)
    wandb.finish()
