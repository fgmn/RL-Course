import torch
import gymnasium as gym
from PPO_gymnasium import PPOContinuous

print('loading pth...')
best_model_path = "best_model.pth"
best_model = torch.load(best_model_path)

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
print("#device:", device)

env_name = "Ant-v4"
env = gym.make(env_name, render_mode="human")
obs, info = env.reset(seed=0)
print('infering...')

torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

agent.actor.load_state_dict(best_model['actor_state_dict'])
agent.critic.load_state_dict(best_model['critic_state_dict'])

state, info = env.reset()

for _ in range(6666):
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    action = agent.take_action(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()

env.close()

