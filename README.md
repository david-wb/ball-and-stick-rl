# ball-and-stick-rl

<img src="static/ball_and_stick.gif" alt="Viewer" width="1000">

# Overview

This repo uses the SAC algorithm to train an agent to balance an inverted pendulum riding on top of a rolling sphere. The agent must
control the xy torques on the sphere to keep the pendulum upright, while also maintaining a target linear velocity.

# Setup

Install dependencies with `poetry`

```bash
poetry install
```

# Training

Launch training with

```bash
./train_sac.sh
```

# Testing

To visualize a trained model in MuJoCo run

```bash
./test_sac.sh
```

# To Launch Viewer

To launch viewer the ball-and-stick model imported run

```bash
./viewer.sh
```

# Notes

The PPO algorithm was also implemented but it does not work well (as implemented anyway).

# Training Metrics

The training metrics are logged to https://wanb.ai

For SAC they look something like this:

<img src="static/wandb.png" alt="SAC Metrics" width="600">
