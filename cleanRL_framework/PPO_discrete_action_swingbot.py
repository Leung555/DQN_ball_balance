# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from sympy import true
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym import spaces
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

Target_Distance = 1.351115 # unit decimeter (dm) # front: 0.5, middle pos: 13.51115, end pos, 27

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hangbot-Cylinder-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class CoppeliaSimEnv(gym.Env):
    def __init__(self):
        super(CoppeliaSimEnv, self).__init__()

        # Setup RemoteAPIClient for CoppeliaSim communication
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        # Load the scene
        # self.sim.loadScene('C:/Users/binggwong/Documents/GitHub/DQN_ball_balance/Dqn_Ball_balance.ttt') # Window
        self.sim.loadScene('/home/binggwong/git/DQN_ball_balance/scenes/Hangbot_cy_2D.ttt') # Ubuntu

        # Get object handles
        self.magnet_sensor_1_handle = self.sim.getObject(":/sensor_m1")
        self.magnet_sensor_2_handle = self.sim.getObject(":/sensor_m2")
        self.magnet1State = true
        self.magnet2State = true


        self.jointHandles = []
        for i in range(1, 7):
            self.jointHandles.append(self.sim.getObject(f":/J{i}"))

        # Curvature Control
        self.Curv = 0.0
        # linear function parameters (y= slope * Curv + const)
        self.slope = 1.0
        self.const = 0

        # body handle
        self.robot_handle = self.sim.getObject(":/Cylinder")
        self.m2_position = [0,0,0]


        # Define action and observation space
        # Action space: 1 continuous + 2 discrete (each discrete: 0 or 1)
        self.action_space = gym.spaces.Dict({
            'continuous': gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32),
            'discrete_1': gym.spaces.Discrete(2),
            'discrete_2': gym.spaces.Discrete(2),
        })
        # Observation space: [curvature, linearVelocity(x, y, z), angularVelocity(x, y, z)]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
            high=np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            dtype=np.float32
        )

        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space

        # Set initial values for simulation state
        self.sim.setStepping(True)
        self.sim.startSimulation()
        self.sim.step()
        self.dt = self.sim.getSimulationTimeStep()

        # Define state
        self.num_states = 9  # [ball_position, ball_velocity, linearVelocity(x, y, z), angularVelocity(x, y, z)]
        self.state = np.zeros(self.num_states)  # [ball_position, ball_velocity]
        self.done = False

    def reset(self, seed=None):
        # Reset the simulation and state
        if seed is not None:
            np.random.seed(seed)        
        # self.sim.stopSimulation()
        # self.sim.setStepping(True)
        # while self.sim.getSimulationState()!= self.sim.simulation_stopped:
        #     pass        
        # self.sim.startSimulation()
        # self.sim.step()
        self.state = np.zeros(self.num_states)
        self.done = False
        return self.state

    def step(self, action):
        # Take action: map action to platform tilting
        # Action is now a dict with 'continuous', 'discrete_1', 'discrete_2'

        position = self.sim.getObjectPosition(self.robot_handle, -1)
        self.m2_position = self.sim.getObjectPosition(self.magnet_sensor_2_handle, -1)
        # print("m2_position: ", self.m2_position)
        linearVelocity, angularVelocity = self.sim.getObjectVelocity(self.robot_handle)
        # print(linearVelocity, angularVelocity)

        # Curvature Control
        # print("Action: ", action)
        # Handle both tensor and numpy array formats
        if isinstance(action['continuous'], np.ndarray):
            self.Curv = float(action['continuous'][0] if len(action['continuous'].shape) > 0 else action['continuous'])
        else:
            self.Curv = float(action['continuous'])
            
        self.magnet1State = bool(action['discrete_1'])  # Discrete action for magnet 1
        self.magnet2State = bool(action['discrete_2'])  # Discrete action for magnet 2
        
        target_jointPos = self.slope * self.Curv + self.const

        # Step simulation
        self.sim.step()

        # Set joint positions
        for i in range(6):
            self.sim.setJointTargetPosition(self.jointHandles[i], target_jointPos)
        
        # magnetic control
        if self.magnet1State:
            self.sim.setStringSignal('magnet1State', 'true')  # Send string data
        else:
            self.sim.setStringSignal('magnet1State', 'false')  # Send string data
        
        if self.magnet2State:
            self.sim.setStringSignal('magnet2State', 'true')  # Send string data
        else:
            self.sim.setStringSignal('magnet2State', 'false')  # Send string data

        # magnet sensor reading
        magnet2_detect = False
        received_magnet2_Data = self.sim.getStringSignal('magnet2_detect')
        if received_magnet2_Data == 'true':
            magnet2_detect = True
        elif received_magnet2_Data == 'false':
            magnet2_detect = False
        # print("magnet2_detect: ", magnet2_detect)

        # Get new state from sensor data
        next_state = np.array([action['continuous'][0], self.magnet1State, self.magnet2State,  # Use the continuous action as the first state variable
                                linearVelocity[0], linearVelocity[1], linearVelocity[2], 
                               angularVelocity[0], angularVelocity[1], angularVelocity[2]])

        # Calculate reward
        reward = self.compute_reward(next_state)

        # Check if the episode is done
        done = abs(position[2]) < 0.55  # Done if the ball goes out of bounds
        terminations = np.array([done])  # Termination condition
        truncations = np.array([False])  # Set to True if you have specific truncation logic
        infos = {}  # You can add any additional information here

        # Update the state
        self.state = next_state

        return next_state, reward, terminations, truncations, infos

    def compute_reward(self, state):
        # Penalize the ball being far from the center
        # ball_pos, ball_vel = state

        target_fw = 0.022
        target_z = 0.756
        y_diff = target_fw - self.m2_position[1]
        z_diff = target_z - self.m2_position[2]
        # print("self.m2_position: ", self.m2_position[1])
        # print("self.m2_position: ", self.m2_position[2])
        reward = - (y_diff ** 2 + z_diff ** 2)  # Reward is negative distance from target position
        # print("reward: ", reward)

        return reward

    def read_magenet_sensor(self):
        # Read proximity sensor data
        res, dist, _, _, _ = self.sim.readProximitySensor(self.ir_sensor_handle)
        if res > 0:
            return dist * 10  # Convert distance to decimeters
        return 0

    def render(self, mode='human'):
        pass  # Rendering can be handled in CoppeliaSim's GUI

    def close(self):
        self.sim.stopSimulation()


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor heads for different action types
        # Continuous action head (for curvature)
        self.actor_continuous_mean = layer_init(nn.Linear(64, 1), std=0.01)
        self.actor_continuous_logstd = nn.Parameter(torch.zeros(1))
        
        # Discrete action heads (for magnets)
        self.actor_discrete_1 = layer_init(nn.Linear(64, 2), std=0.01)  # magnet 1
        self.actor_discrete_2 = layer_init(nn.Linear(64, 2), std=0.01)  # magnet 2

    def get_value(self, x):
        features = self.shared_net(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.shared_net(x)
        
        # Continuous action (curvature)
        continuous_mean = self.actor_continuous_mean(features)
        continuous_std = torch.exp(self.actor_continuous_logstd)
        continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        
        # Discrete actions (magnets)
        discrete_1_logits = self.actor_discrete_1(features)
        discrete_2_logits = self.actor_discrete_2(features)
        discrete_1_dist = Categorical(logits=discrete_1_logits)
        discrete_2_dist = Categorical(logits=discrete_2_logits)
        
        if action is None:
            # Sample actions
            continuous_action = continuous_dist.sample()
            # discrete_1_action = discrete_1_dist.sample()
            discrete_1_action = torch.tensor(1.0, device=features.device)
            discrete_2_action = discrete_2_dist.sample()
            
            action = {
                'continuous': continuous_action,
                'discrete_1': discrete_1_action,
                'discrete_2': discrete_2_action
            }
        else:
            continuous_action = action['continuous']
            discrete_1_action = action['discrete_1']
            discrete_2_action = action['discrete_2']
        
        # Calculate log probabilities
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(axis=-1)
        discrete_1_log_prob = discrete_1_dist.log_prob(discrete_1_action)
        discrete_2_log_prob = discrete_2_dist.log_prob(discrete_2_action)
        
        # Total log probability (sum of independent components)
        total_log_prob = continuous_log_prob + discrete_1_log_prob + discrete_2_log_prob
        
        # Calculate entropy
        continuous_entropy = continuous_dist.entropy().sum(axis=-1)
        discrete_1_entropy = discrete_1_dist.entropy()
        discrete_2_entropy = discrete_2_dist.entropy()
        total_entropy = continuous_entropy + discrete_1_entropy + discrete_2_entropy
        
        return action, total_log_prob, total_entropy, self.critic(features)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )

    envs = CoppeliaSimEnv()
    print("envs.action_space: ", envs.action_space)
    print("envs.single_action_space: ", envs.single_action_space)
    print("envs.observation_space: ", envs.observation_space)
    print("envs.single_observation_space: ", envs.single_observation_space)
    
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"



    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # Store actions as separate tensors for each component
    actions_continuous = torch.zeros((args.num_steps, args.num_envs, 1)).to(device)
    actions_discrete_1 = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    actions_discrete_2 = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(seed=args.seed)
    print("next_obs: ", next_obs)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        print(f"iteration: {iteration}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            # Store action components separately
            actions_continuous[step] = action['continuous']
            actions_discrete_1[step] = action['discrete_1']
            actions_discrete_2[step] = action['discrete_2']
            logprobs[step] = logprob

            # Convert action to numpy format for environment
            action_numpy = {
                'continuous': action['continuous'].cpu().numpy().flatten(),
                'discrete_1': action['discrete_1'].cpu().numpy().item(),
                'discrete_2': action['discrete_2'].cpu().numpy().item()
            }
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action_numpy)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        # Flatten action components separately
        b_actions_continuous = actions_continuous.reshape(-1, 1)
        b_actions_discrete_1 = actions_discrete_1.reshape(-1)
        b_actions_discrete_2 = actions_discrete_2.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Reconstruct action dictionary for the minibatch
                mb_actions = {
                    'continuous': b_actions_continuous[mb_inds],
                    'discrete_1': b_actions_discrete_1[mb_inds],
                    'discrete_2': b_actions_discrete_2[mb_inds]
                }
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], mb_actions)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        print("value_loss:", v_loss.item(), "   ")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()