# ball-and-stick-rl

<img src="static/robot_on_ball.gif" alt="Robot on Sphere" width="1000">

# Overview

This repo uses the SAC algorithm to train an robot to balance on top of a rolling sphere while tracking a target velocity. The robot is essentially an inverted pendulum with three omni-directional wheels in contact with the sphere. The agent must control the three motor torques to keep the pendulum upright and track the target velocity.

# Setup

This currently repo depends on a [fork of MuJoCo](https://github.com/david-wb/mujoco) which contains a small change to support anisotropic friction for the omni-wheels in contact with the sphere. You'll first need to clone and build the fork, including the python bindings, and then install them into the poetry environment of this repo with `poetry add <path to python sdist>`.

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

To launch the MuJoCoviewer and imported the ball-and-stick, run

```bash
./viewer.sh
```

# Training Metrics

The training metrics are logged to https://wanb.ai

For SAC they look something like this:

<img src="static/training_metrics.png" alt="SAC Metrics" width="600">

# Notes

The PPO algorithm was also tested but it did not work well (as implemented anyway).
The SAC algorithm does learn to balance but current plateaus at a sub-optimal performance level
and has difficulty learning to track the target velocity.
