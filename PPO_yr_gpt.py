import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU加速训练
print(f"device: {device}")
directory = './expur5e.'

# actor: 输入为状态，连续动作空间用神经网络生成动作的分布
class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_action):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim) # 共享隐藏层同时输出动作分布的均值和标准差
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.max_action * torch.tanh(self.fc_mu(x)) # 通过tanh将均值控制在范围内
        std = F.softplus(self.fc_std(x)) + 1e-4    # softplus确保正数
        return mu, std

# critic：输入为状态，输出为状态的价值，相当于用神经网络拟合一个值函数指导actor的策略更新
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # 只有一层隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # 隐藏层输出后引入非线性
        return self.fc2(x)

class PPO:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, action_dim, max_action, 
                 hidden_dim=256, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lmbda=0.95, epochs=10, eps=0.2,
                 device='cuda', grad_clip=0.5):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, max_action).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 超参数
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.grad_clip = grad_clip # 新增梯度裁剪防止梯度爆炸

        # 训练记录
        self.train_info = {
            'actor_loss': [],
            'critic_loss': []
        }

    def select_action(self, state, deterministic=False):
        """获取动作（训练时采样，测试时使用均值）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, std = self.actor(state)
        if deterministic:
            return mu.cpu().numpy().flatten()
        dist = Normal(mu, std)
        action = dist.sample()
        return action.cpu().numpy().flatten()


    # 广义优势估计GAE：采用单步TD低方差高偏差，蒙特卡洛高方差低偏差，GAE指数加权多步TD误差lmbda=0为单步TD，lmbda=1蒙特卡洛
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy() # 断开梯度
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]: # 逆序遍历
            advantage = gamma * lmbda * advantage + delta # 递归计算
            advantage_list.append(advantage)
        advantage_list.reverse()  # 恢复原始顺序
        return torch.tensor(advantage_list, dtype=torch.float)
                
    def update(self, states, actions, old_log_probs, advantages, returns):
        """执行PPO更新"""
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages = advantages.detach()
        returns = returns.detach()
        
        # 多轮次优化
        for _ in range(self.epochs):
            # 计算新策略的概率
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            
            # 策略损失
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            values = self.critic(states).view(-1)
            critic_loss = F.mse_loss(values, returns)
            
            # 梯度更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # 记录损失
            self.train_info['actor_loss'].append(actor_loss.item())
            self.train_info['critic_loss'].append(critic_loss.item())

