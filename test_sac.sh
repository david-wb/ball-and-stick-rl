#!/bin/bash

# Detecting the operating system
OS=$(uname -s)

# Running the command based on the OS
if [ "$OS" = "Darwin" ]; then
    # macOS
    poetry run mjpython -m ball_and_stick_rl.test_sac
else
    # Linux (and others)
    poetry run python -m ball_and_stick_rl.test_sac
fi