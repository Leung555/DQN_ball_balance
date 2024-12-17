# Ball Balance on 1D Lever Problem

<div style="text-align: center;">
    <img src="docs/Animation_ball_balance_PPO.gif" alt="Ball Balance GIF" width="300" />
</div>
This project implements reinforcement learning algorithms, specifically DQN (Deep Q-Network) and PPO (Proximal Policy Optimization), to solve the ball balancing task on a 1D lever. The objective is to keep the ball centered on the lever by applying appropriate tilting actions.

## Overview

The simulation is conducted using CoppeliaSim (formerly V-REP), which allows for realistic physics simulations of the ball and lever dynamics. The environment is designed to train agents using deep reinforcement learning techniques, where they learn to take actions that maximize their rewards while keeping the ball balanced.

## Features

- **Algorithms Implemented:** DQN and PPO
- **Environment:** CoppeliaSim for physics simulation
- **State Representation:** The state is represented by the ball's position and velocity.
- **Action Space:** Discrete actions to tilt the lever left, right, or stay still.
- **Customizable Hyperparameters:** Easily adjustable parameters for training the models.

## Getting Started

### Prerequisites

- CoppeliaSim installed with appropriate scene file for the ball balance task.
- Python 3.8-3.11, of currently you are not using python version 3.8-3.11, you can use `pyenv` to install a new python version and create an isolated virtual environment.
- dependencies installation similar to `cleanRL repository` [link](https://github.com/vwxyzjn/cleanrl/tree/master?tab=readme-ov-file) It Required libraries: `gymnasium`, `torch`, `numpy`, `tyro`, `coppeliasim_zmqremoteapi_client`, and `torch.utils.tensorboard`
    ```
    git clone https://github.com/Leung555/DQN_ball_balance.git && cd DQN_ball_balance
    poetry install

    # alternatively, you could use `poetry shell` and do
    # `python run cleanRL_framework/PPO_discrete_action.py`
    poetry run python cleanRL_framework/PPO_discrete_action.py
    
    # open another terminal and enter `cd cleanrl/cleanrl`
    tensorboard --logdir runs

    ```


