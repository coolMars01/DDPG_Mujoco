import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw
from ur5e_env import UR5eEnv
from backup.DDPG import DDPG


class UR5eSimulator:
    def __init__(self):
        # 文件保存设置
        self.folderpath = "M:/ur5e_DDPG_trajectory_planning_20000csv/"
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
        self.setup_visualization()

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
            float(self.env.action_bound0[1]),
            float(self.env.action_bound1[1]),
            float(self.env.action_bound2[1]),
            float(self.env.action_bound3[1]),
            float(self.env.action_bound4[1]),
            float(self.env.action_bound5[1])
        ]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_action = torch.FloatTensor(max_action_list).to(self.device)
        self.agent = DDPG(self.state_dim, self.action_dim, self.max_action)
        self.load = False
        if self.load:
            self.agent.load()   # 加载训练好的权重文件

        np.random.seed(0)
        torch.manual_seed(0)

        # 画图及数据保存相关变量
        self.plot_timestep = 2000
        self.delta_time = 0.034
        self.episode_list = []
        self.reward_list = []
        self.reward_table_list = []
        self.save_table = True

    @staticmethod
    def mkdir(folderpath, foldername):
        full_path = folderpath + foldername
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f'The folder named {foldername} has been created')
        else:
            print("The folder already exists!")

    def setup_visualization(self):
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
        episode_counter = 0
        t = 0
        dt = self.dt
        while not glfw.window_should_close(self.window) and episode_counter < self.max_episode:
            # 初始化当前回合的数据保存列表
            timelist = []
            steplist = []
            qpos_lists = [[] for _ in range(6)]
            endpointlist = []

            # 重置环境与仿真数据
            self.state = self.env.reset()  # state前6个元素为关节角度，后3为目标位置
            for j in range(6):
                self.data.qpos[j] = self.state[j]
            mj.mj_forward(self.model, self.data)
            mj.mj_step(self.model, self.data)
            total_reward = 0
            step = 0
            done = False
            # 获取末端点和目标点信息
            end_point = {'x': self.data.site_xpos[0][0],
                         'y': self.data.site_xpos[0][1],
                         'z': self.data.site_xpos[0][2]}
            goal = {'x': self.data.site_xpos[1][0],
                    'y': self.data.site_xpos[1][1],
                    'z': self.data.site_xpos[1][2]}
            dist4 = np.array([goal['x'] - end_point['x'],
                              goal['y'] - end_point['y'],
                              goal['z'] - end_point['z']])
            # 将状态扩展为13维
            self.state = np.concatenate((self.state, dist4, [0.]), axis=0)
            episode_counter += 1
            self.episode_list.append(episode_counter)
            target_init = [self.state[6], self.state[7], self.state[8]]

            while not done:
                end_point = {'x': self.data.site_xpos[0][0],
                             'y': self.data.site_xpos[0][1],
                             'z': self.data.site_xpos[0][2]}
                goal = {'x': self.data.site_xpos[1][0],
                        'y': self.data.site_xpos[1][1],
                        'z': self.data.site_xpos[1][2]}
                # 强化学习决策并添加噪声（帮助探索）
                action = self.agent.select_action(self.state)
                action = (action + np.random.normal(0, 0.01, size=6)).clip(
                    self.env.action_bound0[0], self.env.action_bound0[1])
                next_state, reward, done = self.env.step(action, target_init, goal, end_point)

                print(f'末端点的位置为({self.data.site_xpos[0][0]},{self.data.site_xpos[0][1]},{self.data.site_xpos[0][2]})')
                print(f'目标点的位置为({self.data.site_xpos[1][0]},{self.data.site_xpos[1][1]},{self.data.site_xpos[1][2]})')

                if step <= 998:
                    total_reward += reward

                endpointlist.append([self.data.site_xpos[0][0], self.data.site_xpos[0][1],
                                     self.data.site_xpos[0][2], total_reward])

                # 存入经验回放
                self.agent.replay_buffer.push((self.state, next_state, action, reward, np.float32(done)))
                self.state = next_state

                time_prev = t
                while t - time_prev < 1.0 / 60.0:
                    mj.mj_forward(self.model, self.data)
                    t += dt
                    mj.mj_step(self.model, self.data)
                if done:
                    break

                step += 1
                timelist.append(self.data.time)
                steplist.append(step)
                for j in range(6):
                    qpos_lists[j].append(self.data.qpos[j])

                # 更新并渲染场景
                viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                mj.mjr_render(viewport, self.scene, self.context)
                glfw.swap_buffers(self.window)
                glfw.poll_events()

            self.total_step += step + 1
            print(f"Total T:{self.total_step} Episode:{episode_counter} Total Reward:{total_reward:0.2f}")
            self.reward_list.append(total_reward)
            self.agent.update()
            self.reward_table_list.append(np.array([episode_counter, total_reward]))

            # 保存CSV数据
            if episode_counter % 1 == 0 and total_reward >= -5000 and self.save_table:
                end_motion_table = pd.DataFrame(endpointlist, columns=['x', 'y', 'z', 'reward'])
                end_motion_table.to_csv(
                    os.path.join(self.folderpath, self.foldername1, f'end_motion_table_{episode_counter}.csv'),
                    index=False)
                qpos_table = pd.DataFrame(
                    np.column_stack((steplist, np.array(qpos_lists).T, total_reward * np.ones(len(steplist)))),
                    columns=['time', 'shoulder link', 'upper arm link', 'forearm link', 'wrist1 link', 'wrist2 link',
                             'wrist3 link', 'total_reward']
                )
                qpos_table.to_csv(
                    os.path.join(self.folderpath, self.foldername2, f'joint_path_table_{episode_counter}.csv'),
                    index=False)
                reward_table = pd.DataFrame(self.reward_table_list, columns=['episode', 'total_reward'])
                reward_table.to_csv(os.path.join(self.folderpath, self.foldername3, 'reward_episode_table.csv'),
                                    index=False)

            # 绘图
            self.plot_episode(episode_counter, total_reward, timelist, steplist, qpos_lists, endpointlist)
            if episode_counter % 100 == 0:
                self.agent.save(total_reward, episode_counter)
        glfw.terminate()


if __name__ == '__main__':
    sim = UR5eSimulator()
    sim.run()
