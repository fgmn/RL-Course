import torch
import gymnasium as gym
from DDPG_gymnasium import DDPG

print('loading pth...')
best_model_path = "best_model.pth"
best_model = torch.load(best_model_path)

actor_lr = 1e-4 # 3e-4
critic_lr = 1e-4 # 3e-3
num_episodes = 2000
hidden_dim = 512
gamma = 0.98
tau = 0.0005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 128
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "Ant-v4"
env = gym.make(env_name, render_mode="human")
obs, info = env.reset(seed=0)
print('infering...')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值

agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, 
             sigma, actor_lr, critic_lr, tau, gamma, device)

agent.actor.load_state_dict(best_model['actor_state_dict'])
agent.critic.load_state_dict(best_model['critic_state_dict'])

state, info = env.reset()

for _ in range(66666):
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    action = agent.take_action(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()

env.close()

