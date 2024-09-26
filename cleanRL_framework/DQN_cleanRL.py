# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000 # original 500000 
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

class CoppeliaSimEnv(gym.Env):
    def __init__(self):
        super(CoppeliaSimEnv, self).__init__()

        # Setup RemoteAPIClient for CoppeliaSim communication
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        # Load the scene
        self.sim.loadScene('C:/Users/binggwong/Documents/GitHub/DQN_ball_balance/Dqn_Ball_balance.ttt')

        # Get object handles
        self.ir_sensor_handle = self.sim.getObject(":/Proximity_sensor")
        self.joint_handle = self.sim.getObject(":/Revolute_joint")

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 0: tilt left, 1: tilt right, 2: stay still
        self.observation_space = gym.spaces.Box(low=np.array([-50, -10]), high=np.array([50, 10]), dtype=np.float32)
        self.num_envs = Args.num_envs

        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space

        # Set initial values for simulation state
        self.sim.setStepping(True)
        self.sim.startSimulation()
        self.sim.step()
        self.dt = self.sim.getSimulationTimeStep()

        # Define state
        self.state = np.zeros(2)  # [ball_position, ball_velocity]
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
        self.state = np.zeros(2)
        self.done = False
        return self.state

    def step(self, action):
        # Take action: map action to platform tilting
        if action == 0:  # Tilt left
            target_joint_pos_deg = 1
        elif action == 1:  # Tilt right
            target_joint_pos_deg = -1
        else:  # Stay still
            target_joint_pos_deg = 0

        self.sim.setJointTargetPosition(self.joint_handle, math.radians(target_joint_pos_deg))

        # Step simulation
        self.sim.step()

        # Get new state from sensor data
        ball_pos = self.read_ir_sensor()
        ball_vel = ball_pos - self.state[0]  # Estimate velocity
        next_state = np.array([ball_pos, ball_vel])

        # Calculate reward
        reward = self.compute_reward(next_state)

        # Check if the episode is done
        done = abs(ball_pos) > 50  # Done if the ball goes out of bounds
        terminations = np.array([done])  # Termination condition
        truncations = np.array([False])  # Set to True if you have specific truncation logic
        infos = {}  # You can add any additional information here

        # Update the state
        self.state = next_state

        return next_state, reward, terminations, truncations, infos

    def compute_reward(self, state):
        # Penalize the ball being far from the center
        ball_pos, ball_vel = state

        alpha = 1.0  # Weight for position penalty
        beta = 1.0   # Weight for velocity penalty

        # Penalize distance from the center
        position_penalty = alpha * abs(Target_Distance - ball_pos)**2
        # if abs(Target_Distance - ball_pos) < 0.2:
        #     position_penalty = 0

        # Velocity penalty that depends on how far the ball is from the center
        # velocity_penalty = beta * abs(ball_vel)**2

        # Combined reward
        reward = - position_penalty # - velocity_penalty

        return reward

    def read_ir_sensor(self):
        # Read proximity sensor data
        res, dist, _, _, _ = self.sim.readProximitySensor(self.ir_sensor_handle)
        if res > 0:
            return dist * 10  # Convert distance to decimeters
        return 0

    def render(self, mode='human'):
        pass  # Rendering can be handled in CoppeliaSim's GUI

    def close(self):
        self.sim.stopSimulation()


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )

    envs = CoppeliaSimEnv()
    print("envs.action_space: ", envs.action_space)
    print("envs.single_action_space: ", envs.single_action_space)
    print("envs.observation_space: ", envs.observation_space)
    print("envs.single_observation_space: ", envs.single_observation_space)
    print("dtype: ", envs.single_observation_space.dtype)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            print("q_values: ", q_values)
            # actions = torch.argmax(q_values, dim=1).cpu().numpy()
            actions = torch.argmax(q_values).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("value_loss:", loss.item(), "   ")


            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
