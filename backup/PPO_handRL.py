import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# actor: 输入为状态，连续动作空间用神经网络生成动作的分布
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim) # 共享隐藏层同时输出动作分布的均值和标准差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x)) # 通过tanh将均值控制在[-2, 2]，符合环境动作力矩范围
        std = F.softplus(self.fc_std(x))     # softplus确保正数
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

class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device) # ndarray转为tensor
        mu, sigma = self.actor(state)                                          # 自动调用forward计算动作分布
        action_dist = torch.distributions.Normal(mu, sigma)                    # 根据均值和标准差定义正态分布（高斯分布）
        action = action_dist.sample()                                          # 从正态分布中采样一个动作
        return [action.item()]                                                 # 返回动作的标量，减少内存消耗

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # NN输入通常是二维的(batch_size*feature_dim), tansition_dict['actions']为一维数组(形为[batch_size]),通过view(-1, 1)变为[batch_size, 1]符合输入要求
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device) 
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

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2 # 截断范围
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.seed(0)
# print("Observation space:", env.observation_space) # 连续 Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# print("Action space:", env.action_space)           # Box([-2.], [2.], (1,), float32)
# print("State example:", env.reset())               # 随机一个重置状态[ 0.23807192 -0.97124755 -0.7595362 ]
# print("State type:", type(env.reset()))            # numpy.ndarray

torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
# print("state_dim:", state_dim, "action_dim:", action_dim,) # 状态[cos sin dtheta] 动作 力矩

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

# 测试mu std形状
# batch_states = torch.randn(64, 3).to(agent.device)
# mu_batch, std_batch = agent.actor(batch_states)
# print(mu_batch.shape, std_batch)  # torch.Size([64, 1])

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)              # 根据状态采取动作
                    next_state, reward, done, _ = env.step(action) # 环境交互
                    transition_dict['states'].append(state)        # 经验回放：形成轨迹数据，为后续策略更新做准备
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state                             # 更新状态
                    episode_return += reward                       # 累计本回合episode奖励值
                return_list.append(episode_return)
                agent.update(transition_dict)                      # 用本episode收集的轨迹数据更新策略网络（Actor）和价值网络（Critic）
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

return_list = train_on_policy_agent(env, agent, num_episodes)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 21)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()