import torch
import torch.nn.functional as F
import numpy as np
import os, re, glob


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {device}")
directory = './ppo_models'

# actor: 输入为状态，连续动作空间用神经网络生成动作的分布
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_action):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim) # 共享隐藏层同时输出动作分布的均值和标准差
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.max_action * torch.tanh(self.fc_mu(x)) # 通过tanh将均值控制在[-2, 2]，符合环境动作力矩范围
        std = F.softplus(self.fc_std(x))                 # softplus确保正数
        return mu, std

# critic：输入为状态，输出为状态的价值，相当于用神经网络拟合一个值函数指导actor的策略更新
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # 只有一层隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # 隐藏层输出后引入非线性
        return self.fc2(x)

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

class PPO:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, actor_lr=3e-4, critic_lr=1e-3, lmbda=0.95, epochs=10, eps=0.2, gamma=0.99, device='cuda'):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, max_action).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

        # 训练记录
        self.train_info = {
            'actor_loss': [],
            'critic_loss': []
        }

    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device) # ndarray转为tensor
        mu, sigma = self.actor(state)                       # 自动调用forward计算动作分布
        action_dist = torch.distributions.Normal(mu, sigma) # 根据均值和标准差定义正态分布（高斯分布）
        action = action_dist.sample()                       # 从正态分布中采样一个动作
        return action.squeeze().tolist()  # 示例：将形状 (2,) 转为 [0.5, -0.2]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device) 
        # NN输入通常是二维的(batch_size*feature_dim), tansition_dict['actions']为一维数组(形为[batch_size]),通过view(-1, 1)变为[batch_size, 1]符合输入要求
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0                                                       # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)             # 计算TD目标值：当前奖励 + 折扣后的下一状态价值
        td_delta = td_target - self.critic(states)                                            # 计算TD误差：目标值与当前状态价值的差值
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device) # 基于值的DQN采用贪婪策略直接选择价值最大的动作，基于策略的用TD
        
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions) # 计算动作的对数概率 logπ(a_t|s_t),后续计算重要性采样比率

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs) # 先对概率求log再反过来用exp还原：避免了概率直接相除出现数值不稳定
            surr1 = ratio * advantage  # 无约束的策略梯度目标：直接最大化 ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage # 带约束的目标：将radio限制在[1-eps,1+eps]，防止策略更新幅度过大
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # min保守优化目标，取负号是因为优化器默认最小化损失，而我们需要最大化策略收益。
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach())) # Critic的目标是准确预测状态价值，最小化当前Critic预测值与TD目标的差异，提升价值估计的准确性
            self.actor_optimizer.zero_grad()  # 将累计梯度归零
            self.critic_optimizer.zero_grad()
            actor_loss.backward()  # 反向传播NN各参数对loss求梯度
            critic_loss.backward()
            self.actor_optimizer.step()  # 优化器根据梯度更新参数
            self.critic_optimizer.step()

            # 记录损失
            self.train_info['actor_loss'].append(actor_loss.item())
            self.train_info['critic_loss'].append(critic_loss.item())

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