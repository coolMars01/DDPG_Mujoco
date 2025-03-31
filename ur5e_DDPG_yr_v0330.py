'''
1. 本代码训练20000轮大概用了8h，环境交互大概耗时0.44s，网络更新大概0.8s-0.9s
2. 根据episode的reward曲线和每100个episode保存的pth模型可知最佳模型为第5900轮，其中最佳训练轮次在5900～8300
3. DDPG训练轮次多，会有奖励突然爆跌的情况，训练和测试效果不稳定
4. 可能需要调整DDPG参数或者奖励函数，以及查看网络的拟合情况
'''

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw
from ur5e_env_yr import UR5eEnv
from DDPG_yr import DDPG
import time


class UR5eSimulator:
    def __init__(self, visualize=True, save_table=True, plot_joint=True, test_mode=False):
        # 文件保存设置
        self.folderpath = "M:/ur5e_DDPG_trajectory_planning_2000csv/"
        self.foldername1 = 'end_motion_path_csv'
        self.foldername2 = 'joint_path_csv'
        self.foldername3 = 'reward_episode_csv'
        self.mkdir(self.folderpath, self.foldername1)
        self.mkdir(self.folderpath, self.foldername2)
        self.mkdir(self.folderpath, self.foldername3)

        # 解决OMP重复报错
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        plt.rcParams['axes.unicode_minus'] = False

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
        self.max_episode = 20000

        # 强化学习环境和智能体初始化
        self.env = UR5eEnv()
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        # 从环境中获取动作边界，构造max_action
        max_action_list = [
            float(self.env.action_bound[0, 1]),
            float(self.env.action_bound[1, 1]),
            float(self.env.action_bound[2, 1]),
            float(self.env.action_bound[3, 1]),
            float(self.env.action_bound[4, 1]),
            float(self.env.action_bound[5, 1])
        ]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_action = torch.FloatTensor(max_action_list).to(self.device)
        self.agent = DDPG(self.state_dim, self.action_dim, self.max_action)
        
        self.test_mode = test_mode
        if self.test_mode:
            self.agent.load(target_episode=-1)
            self.max_episode = 10     # 测试次数
            print(f"Start testing, test episode num is {self.max_episode}")
        else:
            print(f"Start training, train episode num is {self.max_episode}")
        
        np.random.seed(0)
        torch.manual_seed(0)

        # 画图及数据保存相关变量
        self.plot_timestep = 1000        # 每 plot_timestep 画一个episode奖励变化曲线
        self.delta_time = 0.034
        self.episode_list = []
        self.reward_list = []
        self.reward_table_list = []
        self.save_table = save_table    # 是否保存数据为csv文件
        self.plot_joint = plot_joint    # 是否画关节角度图像        


    @staticmethod
    def mkdir(folderpath, foldername):
        full_path = folderpath + foldername
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f'The folder named {foldername} has been created')
        else:
            print("The folder already exists!")

    def setup_visualization(self):
        if not self.visualize:
            return
        
        # 相机参数配置
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.scene = mj.MjvScene(self.model, maxgeom=10000)

        # 设置相机初始视角
        self.cam.azimuth = 118.0    # 方位角（水平旋转）
        self.cam.elevation = -52.8  # 俯仰角（垂直倾斜）
        self.cam.distance = 2.84    # 观察距离
        self.cam.lookat = np.array([-0.0241, 0.01092, 0.24126]) # 焦点坐标

        # 初始化GLFW->创建窗口->激活OpenGL上下文->创建MuJoCo渲染上下文
        if not glfw.init():
            raise Exception("GLFW初始化失败！")
        self.window = glfw.create_window(800, 600, "DDPG", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW窗口创建失败！")
        glfw.make_context_current(self.window) # 将新窗口绑定到当前线程的OpenGL上下文
        glfw.swap_interval(1)  # 开启垂直同步，限制帧率与显示器刷新率同步，设为0则为无限帧率，GPU占用高
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value) # mujoco渲染字体为150%

        # 注册回调函数
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)

        # 设置GLFW鼠标相关状态
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # 设置显示参数
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False # 碰撞接触点显示选项
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False # 碰撞接触力显示选项
        self.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False  # 是否透明选项
        self.model.vis.scale.contactwidth = 0.1                  # 接触面显示宽度
        self.model.vis.scale.contactheight = 0.01                # 接触面显示高度
        self.model.vis.scale.forcewidth = 0.05                   # 力显示宽度
        self.model.vis.map.force = 0.5                           # 力显示长度
        # self.opt.frame = mj.mjtFrame.mjFRAME_GEOM                # 显示geom的frame

    def init_controller(self):
        # 针对六个关节均开启力矩伺服器
        for actuator_no in range(6):
            self.set_torque_servo(actuator_no, 1)

    def set_torque_servo(self, actuator_no, flag):
        if flag == 0:
            self.model.actuator_gainprm[actuator_no, 0] = 0 # 关闭执行器
        else:
            self.model.actuator_gainprm[actuator_no, 0] = 1 # 开启执行器

    def controller(self, model, data):
        # 简单PD控制器，使用self.state作为目标状态（前6个元素为关节角度）
        data.ctrl[0] = -3500 * (data.qpos[0] - self.state[0]) - 100 * (data.qvel[0])
        data.ctrl[1] = -3500 * (data.qpos[1] - self.state[1]) - 100 * (data.qvel[1])
        data.ctrl[2] = -3500 * (data.qpos[2] - self.state[2]) - 100 * (data.qvel[2])
        data.ctrl[3] = -3000 * (data.qpos[3] - self.state[3]) - 100 * (data.qvel[3])
        data.ctrl[4] = -3000 * (data.qpos[4] - self.state[4]) - 100 * (data.qvel[4])
        data.ctrl[5] = -3000 * (data.qpos[5] - self.state[5]) - 100 * (data.qvel[5])

    # GLFW回调函数
    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos
        if not (self.button_left or self.button_middle or self.button_right):
            return
        width, height = glfw.get_window_size(window)
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

    def plot_episode(self, episode, total_reward, timelist, steplist, qpos_lists, endpointlist):
        # 画出回合奖励曲线
        if episode % self.plot_timestep == 0:
            fig = plt.figure(figsize=(9, 6))
            plt.title('Rewards vary with episodes', fontsize=16)
            plt.xlabel('Episodes', fontsize=14)
            plt.ylabel('Rewards', fontsize=14)
            plt.plot(self.episode_list, self.reward_list, linewidth=2, color='C0')
            filename = f'./reward_episode_png/reward_episode_{episode}_{total_reward:0.0f}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close(fig)

        if self.plot_joint:
            # 画出当前回合内各关节角度变化曲线
            fig, ax = plt.subplots(3, 2, figsize=(8, 8))
            xplt = np.array(steplist)
            titles = ['shoulder link', 'upper arm link', 'forearm link', 'wrist1 link', 'wrist2 link', 'wrist3 link']
            for j in range(6):
                r = j // 2
                c = j % 2
                ax[r][c].plot(xplt, np.array(qpos_lists[j]),
                            label=f'N_epi={episode}, R_epi={total_reward:0.0f}')
                ax[r][c].set_title(titles[j])
                ax[r][c].set_xlabel(f'step/{self.delta_time}s')
                ax[r][c].set_ylabel('rad')
                ax[r][c].legend()
            plt.tight_layout()
            filename = f'./joint_path_png/joint_angles_episode_{episode}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close(fig)

    def run(self):
        total_start = time.time()
        episode_counter = 0 # 记录episode的数量
        t = 0
        dt = self.dt
        while episode_counter < self.max_episode:
            episode_start = time.time()
            if self.visualize and glfw.window_should_close(self.window): # 可视化模式下检查窗口关闭
                break
            
            timelist = []                       # 记录self.data.time来画图
            steplist = []                       # 记录一个episode内的step数量
            qpos_lists = [[] for _ in range(6)] # 关节角
            endpointlist = []                   # 储存末端位置和episode每一step的累积reward

            self.state = self.env.reset()  # state前6个元素为关节角度，后3为目标位置
            self.data.qpos[:] = self.state[:6]
            mj.mj_forward(self.model, self.data) # 更新物理模型

            total_reward = 0
            step = 0
            done = False
            # 获取末端点和目标点信息
            end_point = {'x': self.data.site_xpos[0][0],
                         'y': self.data.site_xpos[0][1],
                         'z': self.data.site_xpos[0][2]}  # xml中定义了attachment_site和tip4这俩<site>站点，因此可以使用site_xpos直接读取相关数据
            goal = {'x': self.data.site_xpos[1][0],
                    'y': self.data.site_xpos[1][1],
                    'z': self.data.site_xpos[1][2]}
            dist = np.array([goal['x'] - end_point['x'],
                              goal['y'] - end_point['y'],
                              goal['z'] - end_point['z']])
            # 将状态扩展为13维
            self.state = np.concatenate((self.state, dist, [0.]), axis=0)
            episode_counter += 1
            self.episode_list.append(episode_counter)
            target_init = [self.state[6], self.state[7], self.state[8]]

            while not done:  # 一次while就是一个episode
                end_point = {'x': self.data.site_xpos[0][0],
                             'y': self.data.site_xpos[0][1],
                             'z': self.data.site_xpos[0][2]}
                goal = {'x': self.data.site_xpos[1][0],
                        'y': self.data.site_xpos[1][1],
                        'z': self.data.site_xpos[1][2]}
                
                if self.test_mode:
                    action = self.agent.select_action(self.state)  # 确定性策略
                else:
                    action = (self.agent.select_action(self.state) + np.random.normal(0, 0.01, size=6)).clip(self.env.action_bound[0, 0], self.env.action_bound[0, 1])     # 添加噪声有几率跳出局部最优解
                
                next_state, reward, done = self.env.step(action, target_init, goal, end_point) # 100次接近目标或耗尽1000轮时间步
                if step <= 998:             # 实验时间步为1000，step:0-999，但是999为终止态不加奖励
                    total_reward += reward  # env.step 只是计算单次时间步的奖励，这里才是累积整个 episode 的奖励

                endpointlist.append([self.data.site_xpos[0][0], self.data.site_xpos[0][1], self.data.site_xpos[0][2], total_reward])
                self.agent.replay_buffer.push((self.state, next_state, action, reward, np.float32(done)))
        
                # 更新环境中的状态和时间步
                self.state = next_state
                mj.mj_step(self.model, self.data)
                t += self.model.opt.timestep    

                if done:
                    break

                step += 1
                timelist.append(self.data.time)
                steplist.append(step)
                for j in range(6):
                    qpos_lists[j].append(self.data.qpos[j])

                # 仅可视化时更新并渲染场景
                if self.visualize:
                    viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                    mj.mjr_render(viewport, self.scene, self.context)
                    glfw.swap_buffers(self.window) 
                    glfw.poll_events()

            self.total_step += step + 1
            episode_time = time.time() - episode_start
            total_time = time.time() - total_start
            print(f"Episode:{episode_counter} Step:{step+1}/{self.total_step} Reward:{total_reward:0.2f} Time:{episode_time:.2f}/{total_time:.2f}s")
            
            self.reward_list.append(total_reward)
            self.agent.update() # env交互0.46s，更新网络0.8-0.9s，基本构成整个训练时间
            self.reward_table_list.append(np.array([episode_counter, total_reward])) # 每个episode的总奖励

            # 保存CSV数据
            if self.save_table and episode_counter % 1 == 0 and total_reward >= -5000:
                end_motion_table = pd.DataFrame(endpointlist, columns=['x', 'y', 'z', 'reward']) # 这个reward为一个episode每一个step对应的累积reward
                end_motion_table.to_csv(os.path.join(self.folderpath, self.foldername1, f'end_motion_table_{episode_counter}.csv'), index=False)
                
                qpos_table = pd.DataFrame(
                    np.column_stack((steplist, np.array(qpos_lists).T, total_reward * np.ones(len(steplist)))),
                    columns=['time', 'shoulder link', 'upper arm link', 'forearm link', 'wrist1 link', 'wrist2 link',
                             'wrist3 link', 'total_reward']
                )
                qpos_table.to_csv(os.path.join(self.folderpath, self.foldername2, f'joint_path_table_{episode_counter}.csv'), index=False)
                
                reward_table = pd.DataFrame(self.reward_table_list, columns=['episode', 'total_reward'])
                reward_table.to_csv(os.path.join(self.folderpath, self.foldername3, 'reward_episode_table.csv'), index=False)

            self.plot_episode(episode_counter, total_reward, timelist, steplist, qpos_lists, endpointlist)
            if episode_counter % 100 == 0:
                self.agent.save(total_reward, episode_counter)
        
        if self.visualize:
            glfw.terminate()


if __name__ == '__main__':
    sim = UR5eSimulator(visualize=True, save_table=False, plot_joint=False, test_mode=False)
    sim.run()
