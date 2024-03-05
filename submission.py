import torch
from PPO_gymnasium import PPOContinuous

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


torch.manual_seed(0)
state_dim = 17
action_dim = 6  # 连续动作空间

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

agent.actor.load_state_dict(best_model['actor_state_dict'])
agent.critic.load_state_dict(best_model['critic_state_dict'])


def my_controller(dict):
    obs = dict['obs']
    # controlled_player_index = dict['controlled_player_index']

    action = agent.take_action(obs)
    return action

