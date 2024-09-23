import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Constants
SIM_TIME_LENGTH = 10000000.0
EP_LENGTH = 500
sim_step = 0

# Hyperparameters
STATE_SIZE = 2  # Ball position and velocity
ACTION_SIZE = 1  # Continuous action (1 joint position)
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON_CLIP = 0.2  # Clipping range for PPO
K_EPOCHS = 4  # Number of epochs for each PPO update
ENTROPY_BETA = 0.01  # Entropy coefficient for exploration

def is_valid_state(state):
    return not np.any(np.isnan(state)) and not np.any(np.isinf(state))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_size)
        self.sigma = nn.Linear(64, action_size)
        self.value = nn.Linear(64, 1)
        self.apply(init_weights)  # Apply the custom initialization

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = torch.clamp(self.sigma(x),  min=1e-3, max=2)
        value = self.value(x)
        return mu, sigma.exp(), value

    def get_action(self, state):
        mu, sigma, _ = self.forward(state)
        mu, sigma, _ = self.forward(state)

        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            print("NaN detected in mu or sigma:")
            print("mu:", mu)
            print("sigma:", sigma)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()  # Sample from Normal distribution
        log_prob = dist.log_prob(action).sum()
        return action, log_prob

    def get_value(self, state):
        _, _, value = self.forward(state)
        return value
    
# Reward function
def compute_reward(ball_pos, ball_vel):
    alpha = 1.0  # Weight for position penalty
    beta = 1.0   # Weight for velocity penalty
    Target_Distance = 2.7  # Target distance from the center
    position_penalty = alpha * abs(Target_Distance - ball_pos) ** 2
    reward = -position_penalty  # Simplified reward

    return reward

# PPO Agent class
class PPOAgent:
    def __init__(self, state_size, action_size, lr=LEARNING_RATE):
        self.policy = ContinuousPolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        self.clear_memory()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))

    def clear_memory(self):
        self.memory = []

    def compute_returns_and_advantages(self, next_value):
        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(self.memory))):
            state, action, log_prob, reward, value, done = self.memory[step]
            if step == len(self.memory) - 1:
                next_return = next_value
            else:
                next_return = returns[0]
            td_error = reward + GAMMA * next_return * (1 - done) - value
            gae = td_error + GAMMA * 0.95 * gae * (1 - done)
            returns.insert(0, gae + value)
            advantages.insert(0, gae)
        return returns, advantages

    def update(self):
        states, actions, log_probs, rewards, values, dones = zip(*self.memory)
        
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        log_probs_tensor = torch.FloatTensor(log_probs)
        values_tensor = torch.FloatTensor(values)

        next_value = self.policy.get_value(states_tensor[-1].unsqueeze(0)).item()
        returns, advantages = self.compute_returns_and_advantages(next_value)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)

        for _ in range(K_EPOCHS):
            new_log_probs = []
            new_values = []
            entropies = []
            
            for i in range(len(states_tensor)):
                action, log_prob = self.policy.get_action(states_tensor[i].unsqueeze(0))
                new_log_probs.append(log_prob)
                new_values.append(self.policy.get_value(states_tensor[i].unsqueeze(0)))

                dist = torch.distributions.Normal(action, 1.0)
                entropy = dist.entropy()
                entropies.append(entropy)

            new_log_probs = torch.stack(new_log_probs).squeeze()
            new_values = torch.stack(new_values).squeeze()
            entropies = torch.stack(entropies).squeeze()

            ratios = torch.exp(new_log_probs - log_probs_tensor)
            surr1 = ratios * advantages_tensor
            surr2 = torch.clamp(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns_tensor - new_values).pow(2).mean()
            entropy_loss = -entropies.mean()

            loss = actor_loss + 0.5 * critic_loss + ENTROPY_BETA * entropy_loss

            # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.clear_memory()

# Main PPO Training Loop
def ppo_train_loop():
    # Setup remote API and simulation environment
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()

    ir_sensor_handle = sim.getObject("/Proximity_sensor")
    joint_handle = sim.getObject("/Revolute_joint")

    ppo_agent = PPOAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    episodes = 100
    total_rewards_per_episode = []

    for episode in range(episodes):
        state = np.zeros(STATE_SIZE)
        total_reward = 0
        sim_step = 0

        while sim_step < 500:
            ball_pos = read_IR(ir_sensor_handle)
            ball_vel = ball_pos - state[0] if state[0] >= 0 else 0
            state = np.array([ball_pos, ball_vel])

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if not is_valid_state(state):
                print(f"Invalid state encountered at step {sim_step}: {state}")
                break  # Stop the episode if invalid state

            action, log_prob = ppo_agent.policy.get_action(state_tensor)
            value = ppo_agent.policy.get_value(state_tensor)

            clipped_action = torch.clamp(action, min=-2, max=2).item()
            sim.setJointTargetPosition(joint_handle, math.radians(clipped_action))
            sim.step()

            next_ball_pos = read_IR(ir_sensor_handle)
            next_ball_vel = next_ball_pos - ball_pos
            next_state = np.array([next_ball_pos, next_ball_vel])

            reward = compute_reward(next_ball_pos, next_ball_vel)
            done = abs(next_ball_pos) > 50

            ppo_agent.store_transition(state, action.item(), log_prob.item(), reward, value.item(), done)
            state = next_state
            total_reward += reward

            sim_step += 1

        ppo_agent.update()
        total_rewards_per_episode.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    sim.stopSimulation()
    plt.ioff()
    plt.show()


# Helper function to read from the proximity sensor
def read_IR(ir_sensor_handle):
    res, dist, point, obj, n = sim.readProximitySensor(ir_sensor_handle)
    if res > 0:
        return dist * 10  # Convert to decimeters (range 0-3)

if __name__ == "__main__":
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.loadScene("C:/Users/binggwong/Documents/GitHub/DQN_ball_balance" + '/Dqn_Ball_balance.ttt')


    ppo_train_loop()
