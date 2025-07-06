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
        <motor name="torque_x" joint="ball_joint" gear="1 0 0" ctrlrange="-20 20"/>
        <motor name="torque_y" joint="ball_joint" gear="0 1 0" ctrlrange="-20 20"/>
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

        self.action_space = spaces.Box(
            low=-20.0, high=20.0, shape=(2,), dtype=np.float32
        )

        obs_size = 3 + 3 + 3 + 2  # z_axis, ang_vel, base_vel, target_velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.viewer = None
        self.frame_skip = 4
        self.step_count = 0

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
        self.step_count = 0
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
        self.step_count += 1
        if target_velocity is not None:
            self.target_velocity = np.array(target_velocity, dtype=np.float32)

        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        z_axis = obs[:3]  # Normalized z_axis from _get_obs
        angle_deviation = (
            np.arccos(np.clip(z_axis[2], -1, 1)) / np.pi
        )  # Normalize angle deviation to [0, 1]

        upright_reward = 1 - angle_deviation

        base_vel = obs[6:9][:2]  # Normalized base velocity (x, y components)
        vel_error = np.linalg.norm(
            base_vel * self.max_speed - self.target_velocity
        )  # Un-normalize base_vel
        max_vel_error = 2 * self.max_speed  # From velocity_range
        velocity_penalty = -1.0 * (vel_error / max_vel_error)
        control_penalty = -0.01 * np.sum(np.square(action))

        reward = float(2 * upright_reward + velocity_penalty + control_penalty)
        terminated = bool(z_axis[2] < 0.1)  # Relaxed termination condition

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
                "velocity_penalty": velocity_penalty,
                "velocity_error": vel_error,
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
        self.next_hiddens = np.zeros((capacity, seq_len, 1, hidden_dim), dtype=np.float32)
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
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))  # Learnable log_std
        self.action_scale = torch.tensor(20.0)  # Scale to [-20, 20]

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
        alpha=0.3,
        hidden_size=32,
        num_layers=1,
        seq_len=100,
        device="cpu",
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
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
        if self.replay_buffer.size < self.batch_size:
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
        learning_rate=2e-4,
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
