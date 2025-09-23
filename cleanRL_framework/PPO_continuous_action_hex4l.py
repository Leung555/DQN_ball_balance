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
    env_id: str = "Hex-4l"
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
        # self.sim.loadScene('/home/binggwong/git/DQN_ball_balance/scenes/Hangbot_cy_2D.ttt') # Ubuntu
        self.sim.loadScene('/home/binggwong/git/DQN_ball_balance/scenes/SOLMbot_hex_4l_clean.ttt') # Ubuntu

        # --- CHANGE HERE for legged robot ---
        # Get handles for all 12 joints (3 per leg × 4 legs)
        joint_names = [
            "R1_J1", "R1_J2", "R1_J3",
            "R2_J1", "R2_J2", "R2_J3",
            "L1_J1", "L1_J2", "L1_J3",
            "L2_J1", "L2_J2", "L2_J3"
        ]
        self.jointHandles = [self.sim.getObject(f":/{name}") for name in joint_names]

        FR_foot_sensor=self.sim.getObject(":/R1_foot_sensor")
        FL_foot_sensor=self.sim.getObject(":/L1_foot_sensor")
        RR_foot_sensor=self.sim.getObject(":/R2_foot_sensor")
        RL_foot_sensor=self.sim.getObject(":/L2_foot_sensor")
        self.foot_sensor_array = { FR_foot_sensor, RR_foot_sensor, FL_foot_sensor, RL_foot_sensor }

        self.body = self.sim.getObject(':/Body')
        # -------------------------------------

        # --- CHANGE HERE for legged robot ---
        # Action space: 12 continuous actions (normalized to [-1, 1])
        self.action_space = gym.spaces.Box(low=-0.5, high=0.5, shape=(12,), dtype=np.float32)
        # -------------------------------------

        # --- CHANGE HERE for legged robot ---
        # Observation space: orientation (3), joint positions (12), joint velocities (12), foot contacts (4)
        obs_low = np.array(
            [-np.pi]*3 + [-np.pi]*12 + [-10.0]*12 + [0]*4
        )
        obs_high = np.array(
            [np.pi]*3 + [np.pi]*12 + [10.0]*12 + [2]*4
        )
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # -------------------------------------

        self.single_action_space = self.action_space
        print("Single action space:", self.single_action_space)
        self.single_observation_space = self.observation_space

        # Set initial values for simulation state
        self.sim.setStepping(True)
        self.sim.startSimulation()
        self.sim.step()
        self.dt = self.sim.getSimulationTimeStep()

        # Define state
        self.num_states = 31  # 3+12+12+4
        self.state = np.zeros(self.num_states)
        self.done = False

    def reset(self, seed=None):
        print("Resetting the environment...")
        if seed is not None:
            np.random.seed(seed)
        self.sim.stopSimulation()
        while self.sim.getSimulationState()!= self.sim.simulation_stopped:
            time.sleep(1)
            break
        self.sim.startSimulation()
        self.sim.step()
        self.state = np.zeros(self.num_states)
        self.done = False
        return self.state

    def step(self, action):
        # # Accept both dict and flat array for backward compatibility
        # if isinstance(action, dict) and 'continuous' in action:
        #     action_vec = np.asarray(action['continuous']).flatten()
        # else:
        #     action_vec = np.asarray(action).flatten()
        # # Set each joint's target position from action vector (assume action in [-1, 1])
        # print("action:", action['continuous'])
        for i, joint_handle in enumerate(self.jointHandles):
            self.sim.setJointTargetPosition(joint_handle, float(action['continuous'][i]))
        self.sim.step()
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = self._check_termination(obs)
        info = {}
        self.state = obs
        return obs, reward, np.array([done]), np.array([False]), info

    def _get_obs(self):
        # --- CHANGE HERE for legged robot ---
        # Collect orientation, joint positions, velocities, and foot contacts
        orientation = self.sim.getObjectOrientation(self.body, -1)
        joint_positions = [self.sim.getJointPosition(j) for j in self.jointHandles]
        joint_velocities = [self.sim.getJointVelocity(j) for j in self.jointHandles]
        # Compute resultant force magnitude from all foot sensors
        resultant_force = [0.0]*4
        for i, sensor_handle in enumerate(self.foot_sensor_array):
            result, force, torque = self.sim.readForceSensor(sensor_handle)
            if result:  # result==1 means valid reading
                force_magnitude = np.linalg.norm(force)/10.0
                resultant_force[i] = force_magnitude
        # print("Resultant force from all foot sensors:", resultant_force)

        obs = np.array(list(orientation) + joint_positions + joint_velocities + resultant_force)
        # print("obs:", obs)
        return obs
        # -------------------------------------

    def compute_reward(self, obs):
        # --- CHANGE HERE for legged robot ---
        # Example reward: encourage forward movement, penalize energy, encourage stability, foot contact
        # You should design this for your specific task
        # Reward for forward walking: encourage positive forward velocity (x-direction)
        reward = 0.0
        # Assume the robot's body linear velocity in x is available as part of observation (if not, you may need to add it)
        # Here, as a placeholder, use the orientation's pitch (obs[1]) as a proxy for forward movement
        # In practice, you should use the actual forward velocity from simulation
        # Extract linear velocity of the body in the x-axis from CoppeliaSim
        # Get the absolute linear and angular velocity of the body
        abs_lin_vel, abs_ang_vel = self.sim.getObjectVelocity(self.body)
        # Get the orientation quaternion of the body in world frame
        q = self.sim.getObjectQuaternion(self.body, -1)  # [x, y, z, w]
        # Compute the inverse quaternion
        q_inv = [-q[0], -q[1], -q[2], q[3]]
        # Rotate the velocities into the local frame using quaternion multiplication
        # CoppeliaSim's multiplyVector expects (quaternion, vector)
        local_lin_vel = self.sim.multiplyVector(q_inv, abs_lin_vel)
        local_ang_vel = self.sim.multiplyVector(q_inv, abs_ang_vel)


        linear_velocity, _ = self.sim.getObjectVelocity(self.body)
        forward_bonus = local_lin_vel[0]  # x-axis linear velocity
        reward += forward_bonus
        # print("forward_bonus:", forward_bonus)

        # Small penalty for large joint movements (energy penalty)
        # reward -= np.sum(np.square(obs[3:15])) * 0.001

        # Encourage feet to touch the ground (contact sensors)
        # reward += sum(obs[-4:]) * 0.1

        # Penalize if robot falls
        # if self._check_termination(obs):
        #     reward -= 100
        # Example: reward += obs[0]  # forward orientation (placeholder)
        # reward -= np.sum(np.square(obs[3:15])) * 0.001  # penalize joint positions (placeholder)
        # reward += sum(obs[-4:]) * 0.1  # encourage feet to touch ground (placeholder)
        # if self._check_termination(obs):
        #     reward -= 100
        return reward
        # -------------------------------------

    def _check_termination(self, obs):
        # --- CHANGE HERE for legged robot ---
        # Example: terminate if robot falls (orientation too large)
        roll, pitch, yaw = obs[0], obs[1], obs[2]
        # Get the robot's position in the world frame
        position = self.sim.getObjectPosition(self.body, -1)
        x, y = position[0], position[1]
        # print(f"Termination check: roll={roll}, pitch={pitch}, yaw={yaw}, x={x}, y={y}")
        if abs(roll) > 1.5 or abs(pitch) > 1.5:
            return True
        if abs(x) > 1.8 or abs(y) > 1.8:
            return True
        return False
        # -------------------------------------

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
            layer_init(nn.Linear(obs_shape, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 16)),
            nn.Tanh(),
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(16, 1), std=1.0),
        )

        # Actor head for continuous action (12-DoF)
        self.action_dim = envs.single_action_space.shape[0]
        self.actor_continuous_mean = layer_init(nn.Linear(16, self.action_dim), std=0.01)
        self.actor_continuous_logstd = nn.Parameter(torch.zeros(self.action_dim))

    def get_value(self, x):
        features = self.shared_net(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.shared_net(x)
        # Continuous action (12-DoF)
        continuous_mean = self.actor_continuous_mean(features)
        continuous_std = torch.exp(self.actor_continuous_logstd)
        continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        if action is None:
            # Sample actions
            continuous_action = continuous_dist.sample()
            action = {
                'continuous': continuous_action,
            }
        else:
            continuous_action = action['continuous']
        # Calculate log probabilities
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(axis=-1)
        # Calculate entropy
        continuous_entropy = continuous_dist.entropy().sum(axis=-1)
        return action, continuous_log_prob, continuous_entropy, self.critic(features)


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
    actions_continuous = torch.zeros((args.num_steps, args.num_envs, 12)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(seed=args.seed)
    # print("next_obs: ", next_obs)
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
            logprobs[step] = logprob

            # Convert action to numpy format for environment
            action_numpy = {
                'continuous': action['continuous'].cpu().numpy().flatten(),
            }
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action_numpy)
            if terminations[0]:
                print("Terminations: ", terminations)
                next_obs = envs.reset(seed=args.seed)
                print("next_obs after reset: ", next_obs)
            next_done = np.logical_or(terminations, truncations).astype(bool)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # --- FIX: ensure next_obs is a float32 numpy array ---
            next_obs = np.asarray(next_obs, dtype=np.float32)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.tensor(next_done, dtype=torch.bool).to(device)

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
                    nextnonterminal = 1.0 - next_done.float()
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