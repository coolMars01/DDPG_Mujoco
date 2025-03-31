import os
import numpy as np
import torch
import pandas as pd
import mujoco as mj
from mujoco.glfw import glfw
from ur5e_env_yr import UR5eEnv
from PPO_yr import PPO
import time
from torch.distributions import Normal



class UR5eSimulator:
    def __init__(self, visualize=True, test_mode=False):
        # MuJoCo模型初始化
        xml_path = './universal_robots_ur5e/scene.xml'
        dirname = os.path.dirname(__file__)
        self.xml_path = os.path.join(dirname, xml_path)
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)

        # 可视化场景设置
        self.visualize = visualize
        if self.visualize:
            self.setup_visualization()
        else:
            self.window = None

        # 初始化控制器（力矩伺服器开启）
        self.init_controller()
        mj.set_mjcb_control(self.controller)

        # 时间步长与其它仿真参数
        self.dt = 0.001
        self.total_step = 0
        self.max_episode = 4000

        # 强化学习环境和智能体初始化
        self.env = UR5eEnv()
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_action = torch.tensor([
            float(self.env.action_bound[i, 1]) for i in range(6)
        ], device=self.device)
        self.agent = PPO(
            state_dim=self.env.state_dim + 4,  # 原始状态 + 目标差异 + 时间步
            action_dim=self.env.action_dim,
            max_action=self.max_action,
            hidden_dim=256,
            actor_lr=3e-4,
            critic_lr=1e-3,
            gamma=0.99,
            lmbda=0.95,
            epochs=10,
            eps=0.2,
            device=self.device
        )
        
        # 训练参数
        self.batch_size = 2048  # 每批收集的步数
        self.max_episode = 2000

    def collect_trajectory(self):
        """收集单条轨迹数据"""
        states, actions, rewards, dones, next_states = [], [], [], [], []
        state = self.env.reset()
        done = False
        
        while not done:
            # 扩展状态
            end_point = self.data.site_xpos[0]
            goal = self.data.site_xpos[1]
            state_ext = np.concatenate([
                state, 
                goal - end_point,
                [self.data.time]
            ])
            
            # 选择动作
            action = self.agent.take_action(state_ext)
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            
            # 存储transition
            states.append(state_ext)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            # 准备下一个状态
            next_end_point = self.data.site_xpos[0]
            next_state_ext = np.concatenate([
                next_state,
                goal - next_end_point,
                [self.data.time + self.dt]
            ])
            next_states.append(next_state_ext)
            
            # 更新状态
            state = next_state
            mj.mj_step(self.model, self.data)
            
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states
        }

    def process_data(self, trajectory):
        """处理轨迹数据用于PPO更新"""
        # 计算价值估计
        with torch.no_grad():
            states_tensor = torch.tensor(trajectory['states'], dtype=torch.float32).to(self.device)
            values = self.agent.critic(states_tensor).cpu().numpy().flatten()
            
            next_states_tensor = torch.tensor(trajectory['next_states'], dtype=torch.float32).to(self.device)
            next_values = self.agent.critic(next_states_tensor).cpu().numpy().flatten()
        
        # 计算优势
        advantages = self.agent.compute_advantages(
            rewards=trajectory['rewards'],
            dones=trajectory['dones'],
            values=values,
            next_values=next_values
        )
        
        # 计算旧策略的对数概率
        with torch.no_grad():
            mu, std = self.agent.actor(states_tensor)
            dist = Normal(mu, std)
            old_log_probs = dist.log_prob(torch.tensor(trajectory['actions'])).sum(dim=1)
        
        return {
            'states': trajectory['states'],
            'actions': trajectory['actions'],
            'old_log_probs': old_log_probs.cpu().numpy(),
            'advantages': advantages,
            'returns': (advantages + values).detach().cpu().numpy()
        }

    def train(self):
        """主训练循环"""
        for episode in range(self.max_episode):
            # 收集数据
            trajectory = self.collect_trajectory()
            
            # 处理数据
            batch_data = self.process_data(trajectory)
            
            # 更新策略
            self.agent.update(
                states=batch_data['states'],
                actions=batch_data['actions'],
                old_log_probs=batch_data['old_log_probs'],
                advantages=batch_data['advantages'],
                returns=batch_data['returns']
            )
            
            # 保存模型
            if episode % 100 == 0:
                self.agent.save_model(f'./checkpoints/ppo_ur5e_{episode}.pth')
                
            # 打印训练信息
            avg_actor_loss = np.mean(self.agent.train_info['actor_loss'][-self.agent.epochs:])
            avg_critic_loss = np.mean(self.agent.train_info['critic_loss'][-self.agent.epochs:])
            print(f"Ep {episode} | Actor Loss: {avg_actor_loss:.3f} | Critic Loss: {avg_critic_loss:.3f}")

if __name__ == '__main__':
    sim = UR5eSimulator(visualize=False)
    sim.train()