# Make sure to have the add-on "ZMQ remote API" running in
# CoppeliaSim. Do not launch simulation, but run this script

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SIM_TIME_LENGTH = 10000000.0
EP_LENGTH = 500
sim_step = 0

# Neural Network for Q-Learning
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
Target_Distance = 1.351115 # unit decimeter (dm) # front: 0.5, middle pos: 13.51115, end pos, 27
STATE_SIZE = 2  # Ball position and velocity
ACTION_SIZE = 3  # Three actions: tilt left, tilt right, stay still
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor for future rewards
BATCH_SIZE = 512
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
MEMORY_SIZE = 10000  # Replay buffer size
TARGET_UPDATE = 10  # How often to update the target network
EPOCH = 100

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

# Choose action using epsilon-greedy policy
def choose_action(state, epsilon, q_network):
    if np.random.rand() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)  # Random action (exploration)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        q_values = q_network(state_tensor)
        if sim_step % 10 == 0:
            print(f"q_values: {torch.argmax(q_values).item()}")
        return torch.argmax(q_values).item()  # Greedy action (exploitation)

# Update the Q-network
def train(q_network, target_network, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    # Get Q-values for the chosen actions
    q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    # Get the next state Q-values from the target network
    next_q_values = target_network(next_states_tensor).max(1)[0]

    # Compute the target Q-values
    target_q_values = rewards_tensor + GAMMA * next_q_values * (1 - dones_tensor)

    # Compute the loss (mean squared error)
    loss = F.mse_loss(q_values, target_q_values.detach())

    # Update the weights of the Q-network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
# Compute reward (combine position and velocity into one term)
def compute_reward(ball_pos, ball_vel):
    alpha = 1.0  # Weight for position penalty
    beta = 1.0   # Weight for velocity penalty

    # Penalize distance from the center
    position_penalty = alpha * abs(Target_Distance - ball_pos)**2
    if abs(Target_Distance - ball_pos) < 0.5:
        position_penalty = 0

    # Velocity penalty that depends on how far the ball is from the center
    # velocity_penalty = beta * abs(ball_vel)**2 / (1 + abs(Target_Distance - ball_pos))

    # Combined reward
    reward = - position_penalty# - velocity_penalty

    return reward

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode during Training')
    plt.legend()
    plt.show()


# Main DQN Loop
def dqn_train_loop():

    # Initialize plot
    plt.ion()  # Interactive mode on for dynamic plotting
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode during Training')
    line, = ax.plot([], [], label='Total Reward per Episode')
    ax.legend()

    # Initialization
    ir_sensor_handle = sim.getObject(":/Proximity_sensor")  # Handle of the proximity sensor
    joint_handle = sim.getObject(":/Revolute_joint")  # Handle of the proximity sensor
    ball_pos_1 = -1
    ball_pos = -1
    next_ball_pos = -1
    next_ball_vel = -1

    # Initialize Q-network and target network
    q_network = DQNetwork(STATE_SIZE, ACTION_SIZE)
    target_network = DQNetwork(STATE_SIZE, ACTION_SIZE)
    target_network.load_state_dict(q_network.state_dict())  # Synchronize target network
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON

    sim.setStepping(True)
    sim.startSimulation()
    sim.step()

    dt = sim.getSimulationTimeStep()

    # List to store total rewards for each episode
    total_rewards_per_episode = []
    
    # Main loop
    for episode in range(EPOCH):  # Number of episodes for training
        print("--EP. {}--".format(episode))
        state = np.zeros(STATE_SIZE)  # Initialize state
        done = False
        total_reward = 0
        sim_step = 0

        while (t := sim.getSimulationTime()) < SIM_TIME_LENGTH \
                and sim_step < EP_LENGTH \
                and not done:
            
            # print(f'Simulation time: {t:.2f} [s]')
            # print(f'sim_step: {sim_step:.2f}')
            ball_pos = read_IR(ir_sensor_handle)
            ball_vel = ball_pos - ball_pos_1 if ball_pos_1 >= 0 else 0

            state = np.array([ball_pos, ball_vel])

            # Choose an action using epsilon-greedy
            action = choose_action(state, epsilon, q_network)

            # Convert action to joint position
            if action == 0:  # Tilt left
                targetJointPosDeg = 1
            elif action == 1:  # Tilt right
                targetJointPosDeg = -1
            else:  # Stay still
                targetJointPosDeg = 0

            sim.setJointTargetPosition(joint_handle, math.radians(targetJointPosDeg))

            # Step simulation
            sim.step()

            next_ball_pos = read_IR(ir_sensor_handle)
            # print("next_ball_pos: ", next_ball_pos)
            next_ball_vel = next_ball_pos - ball_pos
            # print("next_ball_vel: ", next_ball_vel)
            next_state = np.array([next_ball_pos, next_ball_vel])

            # #### Rewards #################
            reward = compute_reward(next_ball_pos, next_ball_vel)
            # ###############################

            # done = abs(next_ball_pos) > 50  # Example terminal condition: ball falls off lever
            done = abs(next_ball_pos) > 50  # Example terminal condition: ball falls off lever

            # Store transition in memory
            memory.add((state, action, reward, next_state, done))

            # Train the Q-network
            train(q_network, target_network, memory, optimizer)

            # Update state and total reward
            state = next_state
            total_reward += reward

            # Update epsilon for exploration decay
            if epsilon > EPSILON_MIN:
                epsilon *= EPSILON_DECAY
                if sim_step % 50 == 0:
                    print(f"epsilon: {epsilon}")
            
            # Visualize data
            # print(f"reward: {reward}")

            sim_step += 1


        # Update the target network periodically
        if episode % TARGET_UPDATE == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Store the total reward for the current episode
        total_rewards_per_episode.append(total_reward)

        # Update the plot dynamically after each episode
        line.set_xdata(range(len(total_rewards_per_episode)))
        line.set_ydata(total_rewards_per_episode)
        ax.relim()  # Recalculate limits for the new data
        ax.autoscale_view()  # Rescale the view based on updated limits
        plt.pause(0.001)  # Small pause to update the plot

        print(f"Episode {episode}, Total Reward: {total_reward}")

    sim.stopSimulation()

    # After training, keep the final plot displayed
    plt.ioff()  # Turn off interactive mode
    plt.show()

def read_IR(ir_sensor_handle):
    res, dist, point, obj, n = sim.readProximitySensor(ir_sensor_handle) # Read the proximity sensor
    if res > 0:
        # print("dist: ", dist*100) # cm unit
        return dist*10 #unit in decimeter range(0-3)



if __name__ == "__main__":
    # Setup Remote Client for Coppeliasim connection
    client = RemoteAPIClient()
    sim = client.require('sim')
    # sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + '/Dqn_Ball_balance.ttt')
    sim.loadScene("C:/Users/binggwong/Documents/GitHub/DQN_ball_balance" + '/Dqn_Ball_balance.ttt')

    dqn_train_loop()
