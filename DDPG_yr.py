import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import os, re, glob

# ------------------------------------------参数初始化---------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="ur5e") # 决定了tensorboard文件和权重文件存储在什么名称的文件夹里
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient 目标网络软更新系数
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor 目标网络奖励值衰减
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size,
parser.add_argument('--batch_size', default=100, type=int) # mini batch size

# optional parameters
parser.add_argument('--update_iteration', default=100, type=int)
args = parser.parse_args()
# ------------------------------------------参数初始化---------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU加速训练
print(f"device: {device}")
directory = './exp' + args.env_name +'./'

class Replay_buffer():
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size   # 当达到了max_size则用这种方法存储，每次溢出会重头开始存储新的
        else:
            self.storage.append(data)   # 未达到max_size时存储方式

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def x_state(self, t):
        x_s = []
        for i in range(t):
            X, Y, U, R, D = self.storage[i]
            x_s.append(np.array(X, copy=False))
        return np.array(x_s)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state)) #估计值是通过
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            # self.writer.close()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)  #软更新策略网络

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self,total_reward,episode):
        # 保存网络权重 total_reward 和 episode 只是方便命名
        torch.save(self.actor.state_dict(), directory + f'actor{total_reward}_{episode}.pth')
        torch.save(self.critic.state_dict(), directory + f'critic{total_reward}_{episode}.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, target_episode=-1):
        actor_files = glob.glob(os.path.join(directory, 'actor*.pth'))
        if target_episode > 0:
            # 指定episode加载
            print(f"▶▶ Attempting to load specified episode: {target_episode}")
            episode_pattern = re.compile(rf'actor(-?\d+\.\d+)_{target_episode}\.pth')
            matched_files = [f for f in actor_files if episode_pattern.search(f)]
            selected_actor = matched_files[0]
        else:
            # 自动查找最大奖励的模型
            print("▶▶ Auto-selecting best model based on reward")
            model_records = []
            pattern = re.compile(r'actor(-?\d+\.\d+)_(\d+).pth')   # 匹配形如 actor-123_456.pth
            for f in actor_files:
                match = pattern.search(f)
                if match:
                    total_reward = float(match.group(1))
                    episode = int(match.group(2))
                    model_records.append((total_reward, episode, f))
            if not model_records:
                raise FileNotFoundError("No valid model files found")
            
            sorted_models = sorted(model_records, key=lambda x: (-x[0], -x[1])) # 降序
            best_reward, best_episode, selected_actor = sorted_models[0]
            print(f"Selected model: Reward={best_reward}, Episode={best_episode}")
        
        selected_critic = selected_actor.replace('actor', 'critic')
        self.actor.load_state_dict(torch.load(selected_actor))
        self.critic.load_state_dict(torch.load(selected_critic))
        print(f"✓ Successfully loaded:\n"
              f"- Actor: {os.path.basename(selected_actor)}\n"
              f"- Critic: {os.path.basename(selected_critic)}")