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

- Python 3.6+
- Required libraries: `gymnasium`, `torch`, `numpy`, `tyro`, `coppeliasim_zmqremoteapi_client`, and `torch.utils.tensorboard`
- CoppeliaSim installed with appropriate scene file for the ball balance task.

### Installation

1. Clone the repository:

   ```bash
   git clone <https://github.com/Leung555/DQN_ball_balance>
   cd <repository-directory/DQN_ball_balance>
