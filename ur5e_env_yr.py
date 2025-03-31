import numpy as np

class UR5eEnv:
    # 基本参数设置
    dt = 0.03  # 每步时间间隔

    # 每个关节动作范围（每步增量）
    action_bound = np.array([[-0.0189, 0.0189]] * 6)  # 6个关节

    # 每个关节角度限制（以弧度为单位）
    arminfo_bound = np.array([
        [-2 * np.pi, 2 * np.pi],
        [-np.pi, 0],
        [-np.pi / 2, np.pi / 2],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi]
    ])

    state_dim = 13  # 6个关节角 + 3个目标位置 + 3个末端点与目标距离 + 1个是否触碰标志（此处距离向量3维拼接成1个范数也可）
    action_dim = 6  # 6个关节角

    # 固定目标与末端点，实际中目标可以设置为随机，此处保持固定便于调试
    GOAL = {'x': 0.5, 'y': 0.5, 'z': 0.5}
    END_POINT = {'x': 0.5, 'y': 0.5, 'z': 0.5}

    def __init__(self):
        # 用一维数组表示6个关节角度，3个目标位置
        self.arm_info = np.zeros(6, dtype=np.float32)
        self.target_info = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.on_goal = False
        self.on_goal_count = 0

    def reset(self):
        """
        每个 episode 重置时：
          - 将目标位置设置为固定值 [0.5, 0.5, 0.5]
          - 将各关节角度初始化为 0
          - 返回初始状态：关节角度和目标位置拼接成 9 维状态
        """
        self.episode_step = 0
        self.target_info[:] = [0.5, 0.5, 0.5]
        self.arm_info[:] = 0.0
        # 初始状态为6个关节角度和3个目标位置
        state = np.concatenate((self.arm_info, self.target_info))
        return state

    def step(self, action, target_position, goal=None, end_point=None):
        """
        输入：
          - action：形状为 (6,) 的数组，每个分量对应一个关节角度的增量
          - target_position：目标位置，形状为 (3,)
          - goal（可选）：字典形式的目标位置，若未传入则使用 GOAL
          - end_point（可选）：字典形式的末端点位置，若未传入则使用 END_POINT

        过程：
          1. 将关节角度更新为：arm_info = arm_info + action，并剪裁至各自角度范围
          2. 拼接当前关节角和目标位置构成基础状态
          3. 根据提供或默认的目标与末端点计算两者间的距离
          4. 奖励为负的欧式距离，当末端点足够靠近目标（各方向差小于0.02时），额外奖励+10，并累计触碰次数
          5. 当触碰次数超过 100 或步数超过 1000 时，结束该 episode， 即done=True
          6. 最终状态由关节角、目标位置、距离向量以及是否触碰标志拼接而成，共 13 维

        返回：
          - s：下一个状态（13维）
          - reward：奖励值
          - done：布尔值，表示是否结束当前 episode
        """
        self.episode_step += 1
        done = False

        # 更新各关节角度，并剪裁
        self.arm_info += action
        for i in range(6):
            self.arm_info[i] = np.clip(self.arm_info[i], self.arminfo_bound[i, 0], self.arminfo_bound[i, 1])

        # 拼接基础状态（关节角 + 目标位置）
        s = np.concatenate((self.arm_info, target_position))

        # 使用传入的 goal 与 end_point，若未提供则使用默认值
        if goal is None:
            self.goal = self.GOAL
        else:
            self.goal = goal
        if end_point is None:
            self.end_point = self.END_POINT
        else:
            self.end_point = end_point

        # 计算目标与末端点之间的距离向量（3维）
        dist = np.array([
            self.goal['x'] - self.end_point['x'],
            self.goal['y'] - self.end_point['y'],
            self.goal['z'] - self.end_point['z']
        ], dtype=np.float32)

        # 奖励为距离的负值 这里需要仔细研究
        reward = -np.linalg.norm(dist)

        # 若末端点足够接近目标（各方向差均在 0.02 范围内），给予额外奖励并累计触碰次数
        if np.all(np.abs(dist) <= 0.02):
            reward += 10
            self.on_goal = True
            self.on_goal_count += 1
            if self.on_goal_count > 100: # 有100次达到goal
                self.on_goal_count = 0
                done = True

        # 若步数达到上限，结束当前 episode
        if self.episode_step >= 1000:
            done = True

        # 拼接最终状态：基础状态、距离向量、触碰标志（1 表示触碰，0 表示未触碰）
        s = np.concatenate((s, dist, [1.0 if self.on_goal else 0.0]))
        self.on_goal = False
        return s, reward, done

    def sample_action(self):
        return np.random.uniform(self.action_bound[:, 0], self.action_bound[:, 1])


if __name__ == '__main__':
    env = UR5eEnv()
    state = env.reset()
    print("初始目标位置：", env.target_info)
    print(state)
    action = env.sample_action()
    # 没有给出end_point导致step的机器人末端点即为[0.5,0.5,0.5],使得xyz的dist为0，达到目标点on_goal=1
    # 按理说应该是给出六个角度后算正运动学求末端位置
    next_state, reward, done = env.step(action, env.target_info)
    print("下一状态：", next_state)
    print("奖励：", reward, "是否结束：", done)
