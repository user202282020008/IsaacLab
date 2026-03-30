# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 上面这三行是版权声明，表示这段代码的归属和开源协议类型。




# 从 Python 未来的版本中导入 annotations 功能，作用是让类型提示（比如标识一个变量是整数还是字符串）在当前版本也能更顺畅地使用。
from __future__ import annotations

# import 是“导入”的意思，相当于把别人写好的工具箱拿过来用。
# 导入 gymnasium 工具箱，并给它起个简写名字叫 gym。这是一个强化学习的通用标准库，定义了环境的通用格式。
import gymnasium as gym
# 导入 torch 工具箱。这是 PyTorch 深度学习框架的核心，用于进行 GPU 上的高级矩阵（张量）运算。
import torch

# 导入 isaaclab 物理仿真平台下的各种专门工具，起别名方便调用。
import isaaclab.sim as sim_utils
# 导入 Articulation（关节刚体，指代无人机这种有物理属性的模型）和它的配置类。
from isaaclab.assets import Articulation, ArticulationCfg
# 导入 DirectRLEnv（直接强化学习环境基类）及其配置类。继承它们就能快速搭起一个环境。
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# 导入图形界面窗口的基类，用来在屏幕上画出调试面板。
from isaaclab.envs.ui import BaseEnvWindow
# 导入可视化标记，用来在 3D 世界里画辅助线、球体或方块。
from isaaclab.markers import VisualizationMarkers
# 导入交互场景配置，用来定义虚拟世界里有什么东西。
from isaaclab.scene import InteractiveSceneCfg
# 导入底层物理仿真循环的配置（比如每次计算相隔多少毫秒）。
from isaaclab.sim import SimulationCfg
# 导入地形配置，用来生成平地、山地等。
from isaaclab.terrains import TerrainImporterCfg
# 导入 configclass 装饰器，它的作用是把一个普通的 Python 类变成专门用来存放配置数据的“数据类”。
from isaaclab.utils import configclass

# 导入底层数学工具函数：subtract_frame_transforms（计算坐标系相对位置），quat_apply_inverse（四元数逆变换，用于把世界坐标转到自身坐标），quat_apply（四元数正变换）。
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse, quat_apply

# 导入用来做可视化标记的预设配置模板（坐标系标记、球体标记、长方体标记）。
from isaaclab.markers import FRAME_MARKER_CFG, SPHERE_MARKER_CFG, CUBOID_MARKER_CFG

# 导入官方提前建好的 Crazyflie（一种微型开源无人机）的模型配置。
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip 表示告诉代码格式化工具不要移动这一行。

# ==========================================
# 1. 界面控制窗口类：用于在仿真软件里加按钮
# ==========================================
# class 关键字用来定义一个“类”（物体的设计图纸）。继承 BaseEnvWindow。
class QuadcopterEnvWindow(BaseEnvWindow):
    # __init__ 是初始化方法，每次按这个图纸造东西时，最先执行这里。
    # env: QuadcopterEnv 表示传入一个我们下面写的无人机环境；window_name 是窗口的名字，默认叫 "IsaacLab"
    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        # super().__init__ 表示先调用父类（BaseEnvWindow）的初始化方法，打好底子。
        super().__init__(env, window_name)
        # with 语句用来划定一个作用域。这里的意思是：“在主垂直界面的布局里...”
        with self.ui_window_elements["main_vstack"]:
            # “在调试框的布局里...”
            with self.ui_window_elements["debug_frame"]:
                # “在调试专用的垂直布局里...”
                with self.ui_window_elements["debug_vstack"]:
                    # 这一句的作用是：在软件界面上创建一个名字叫 "targets" 的打钩框。
                    # 打钩后，就会在 3D 画面里显示我们设定的绿色球（目标位置）等可视化辅助线。
                    self._create_debug_vis_ui_element("targets", self.env)

# ==========================================
# 2. 环境参数配置类：设定驾校的各种规则和物理参数
# ==========================================
# @configclass 是一个语法糖（装饰器），告诉 Python 这是一个纯用来存参数的类，自动帮我们处理好底层逻辑。
@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # --- 强化学习基础参数 ---
    episode_length_s = 10.0      # episode 指的是“一次尝试”。这里规定无人机每次最多飞 10 秒，10秒后重置。
    decimation = 2               # 降采样率。物理引擎可能每秒算 100 次，但 AI 大脑只每秒做 50 次决定。100/2 = 50。
    action_space = 4             # 动作空间为 4 维：AI 输出 4 个数字来控制（总油门推力、滚转力矩、俯仰力矩、偏航力矩）。
    observation_space = 13       # 观察空间为 13 维：AI 能看到 13 个数字（自身XYZ速度，XYZ旋转速度等）。
    state_space = 0              # 额外状态空间，这里不需要，设为 0。
    debug_vis = True             # 默认开启刚才写的可视化调试功能（显示红绿球）。
    ui_window_class_type = QuadcopterEnvWindow # 绑定上面写好的图形界面。

    # --- Sim2Real (仿真到现实) 核心参数 ---
    # 现实中的无人机有各种缺陷，我们在仿真里故意加入这些缺陷，这样 AI 到现实中才不会懵。
    action_delay_steps = 2         # 动作延迟。AI 下达指令后，故意等 2 个计算步（约 40 毫秒）才执行，模拟真实的信号延迟。
    lin_vel_noise_std = 0.1        # 线性速度的噪音标准差。模拟传感器测不准速度。
    ang_vel_noise_std = 0.2        # 角速度（旋转速度）的噪音标准差。模拟陀螺仪误差。

    # --- 物理引擎底层设置 ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,              # dt 是 Delta Time。物理引擎每 0.01 秒（1/100）计算一次世界的变化。
        render_interval=decimation, # 画面渲染的间隔，跟 AI 做决定的频率保持一致。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",    # 摩擦力计算方式：相乘
            restitution_combine_mode="multiply", # 弹力计算方式：相乘
            static_friction=1.0,                 # 静态摩擦系数（1.0表示很涩，不容易滑）。
            dynamic_friction=1.0,                # 动态摩擦系数。
            restitution=0.0,                     # 弹性系数。0.0 表示掉在地上像块泥巴不会弹起，防止无人机坠地后乱弹。
        ),
    )
    
    # --- 地形设置 ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", # 在 3D 树状结构里的路径名。
        terrain_type="plane",      # 地形类型是 "plane"（无限大的平面）。
        collision_group=-1,        # 碰撞组设为 -1，表示它可以跟任何东西发生物理碰撞。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,                 # 重复设置地面的摩擦力
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,           # 不显示地形的网格辅助线。
    )

    # --- 场景生成设置 ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,           # 并行环境数量。重点：利用 GPU 算力同时运行 4096 架无人机一起训练！
        env_spacing=2.5,         # 每一架无人机之间的间隔是 2.5 米，防止它们互相撞到。
        replicate_physics=True,  # 复制物理属性，提升计算效率。
        clone_in_fabric=True     # 在底层渲染引擎中克隆，提升渲染效率。
    )

    # --- 机器人本体设置 ---
    # 把导入的无人机模型应用到场景里，正则表达式 ".*" 意思是匹配 0 到 4095 号的所有环境。
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9       # 推重比是 1.9。意味着满油门时，推力是无人机自身重力的 1.9 倍，能飞得很快。
    moment_scale = 0.01          # 力矩缩放系数，用于微调旋转时的力道。

    # --- 奖励函数权重系数 ---
    # 引导 AI 学习的指挥棒。正数是奖励，负数是惩罚。
    tracking_lin_vel_reward_scale = 10.0   # 成功追踪设定的直线速度，给极大的奖励（10.0）。
    tracking_yaw_rate_reward_scale = 5.0   # 成功追踪设定的偏航（机头左右转）速度，给中等奖励（5.0）。
    roll_pitch_penalty_scale = -0.05       # 惩罚过大的滚转（左右倾斜）和俯仰（前后倾斜），扣分（-0.05），迫使飞机保持平稳。
    z_pos_reward_scale = 1.0               # 保持在特定的高度范围，给少量奖励（1.0）。


# ==========================================
# 3. 环境主逻辑类：真正负责计算世界运行规律的大脑
# ==========================================
class QuadcopterEnv(DirectRLEnv):
    # 声明 cfg 变量必须是刚才写的 QuadcopterEnvCfg 类型。
    cfg: QuadcopterEnvCfg

    # 环境初始化
    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # 初始化父类，把基本功能准备好。
        super().__init__(cfg, render_mode, **kwargs)

        # torch.zeros 是在 GPU（self.device）上创建一个全都是 0 的矩阵（张量）。
        # 这里创建了一个形状为 [4096, 4] 的大本子，用来记录 4096 架飞机当前要做的 4 个动作。
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # 用来存储物理引擎需要的向上的推力 [4096, 1, 3] (3代表XYZ三个方向的力，推力只在Z方向有值)。
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 用来存储物理引擎需要的旋转力矩 [4096, 1, 3] (XYZ三个轴的旋转力)。
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # 为了模拟前面提到的“动作延迟”，我们需要一个历史记录本。
        # 形状是 [延迟步数+1, 4096架飞机, 4个动作维度]。它像一个传送带，把过去的动作传到当前。
        self._action_history = torch.zeros(self.cfg.action_delay_steps + 1, self.num_envs, 4, device=self.device)
        
        # 教练给下达的“期望速度”目标指令。
        self._desired_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device) # 期望在世界空间下的XYZ飞行速度。
        self._desired_yaw_rate_b = torch.zeros(self.