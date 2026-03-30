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
        self._desired_yaw_rate_b = torch.zeros(self.num_envs, 1, device=self.device) # 期望在自身空间下的左右转速度。

        # 一个字典，用来累计各个维度的得分，方便最后统计这 10 秒内无人机的整体表现。
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["tracking_lin_vel", "tracking_yaw_rate", "roll_pitch_penalty", "z_pos"]
        }
        
        # 找到无人机的主体躯干部分（获取它在物理引擎里的身份证号）。
        self._body_id = self._robot.find_bodies("body")[0]
        # 读取无人机的质量（多重）。
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        # 获取重力加速度的大小（约等于 9.81）。
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        # 计算无人机的标准重力（质量 乘 重力加速度 = 重量/牛顿）。
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # 告诉底层系统：按照配置文件的意思，打开红绿辅助球的可视化功能。
        self.set_debug_vis(self.cfg.debug_vis)

    # 准备 3D 场景
    def _setup_scene(self):
        # 按照之前写的配置实例化机器人
        self._robot = Articulation(self.cfg.robot)
        # 把机器人注册到场景大名单里
        self.scene.articulations["robot"] = self._robot
        # 同步环境数量和间距给地形生成器
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # 关键一步：把 1 个做好的环境，克隆出剩下的 4095 个！
        self.scene.clone_environments(copy_from_source=False)
        
        # 如果你没用显卡而是用了 CPU，设置忽略跟地面的复杂内部碰撞来避免卡顿。
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # 加一个照亮整个世界的大灯泡，否则画面是黑的。
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # 物理计算发生前执行的处理（把 AI 的信号转换为真实的物理力）
    def _pre_physics_step(self, actions: torch.Tensor):
        # clamp 限制动作的范围在 -1.0 到 1.0 之间，防止 AI 瞎输出导致引擎崩溃。
        clamped_actions = actions.clone().clamp(-1.0, 1.0)
        # torch.roll 是把数组里的数据整体挪一个位置。比如 [A, B, C] 变成 [C, A, B]。
        # 这是为了模拟延迟，把新的指令放在传送带第一位，然后挤掉最后一位。
        self._action_history = torch.roll(self._action_history, shifts=1, dims=0)
        self._action_history[0] = clamped_actions
        # 从传送带的最后面拿出被延迟过的旧动作，作为当前真正要执行的动作。
        delayed_actions = self._action_history[-1]
        self._actions = delayed_actions
        
        # 数学转换：算推力。AI输出第一维 actions[:, 0] 在 [-1, 1]。
        # (动作 + 1.0) / 2.0 把它变成了 [0, 1] 之间的比例。
        # 再乘上 无人机重量 和 推重比，就得出了此时真实的向上推力牛顿值。
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # 其他三个维度的动作，乘以缩放系数，直接变成翻滚、俯仰和偏航的旋转力矩。
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    # 真正把力施加到物体上
    def _apply_action(self):
        # 调用底层永久力矩合成器，把刚才算好的推力(forces)和旋转力(torques)狠狠推在无人机躯干(body_ids)上。
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # 收集无人机的观察数据（传感器信息）给 AI 大脑看
    def _get_observations(self) -> dict:
        # 获取无人机真实坐标系下的线速度和角速度。
        real_lin_vel_b = self._robot.data.root_lin_vel_b
        real_ang_vel_b = self._robot.data.root_ang_vel_b
        # 加上配置里的正态分布噪音（torch.randn_like），糊弄一下 AI，模拟现实世界的风吹草动和劣质传感器。
        obs_lin_vel_b = real_lin_vel_b + torch.randn_like(real_lin_vel_b) * self.cfg.lin_vel_noise_std
        obs_ang_vel_b = real_ang_vel_b + torch.randn_like(real_ang_vel_b) * self.cfg.ang_vel_noise_std
        
        # 世界坐标转机体坐标。目标点是在世界里的绝对方向，但对无人机来说，它只需要知道“目标在我的前方还是左方”，所以需要四元数逆变换。
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_lin_vel_w)
        
        # torch.cat 把所有这些信息拼接成一根长长的数据条（13个数字），取名 "policy" 交给 AI 的神经网络。
        obs = torch.cat(
            [
                obs_lin_vel_b,                         # 它以为的自己移动速度
                obs_ang_vel_b,                         # 它以为的自己翻滚速度
                self._robot.data.projected_gravity_b,  # 重力在它自己坐标系下的投影（让它知道自己是不是翻肚皮了）
                desired_lin_vel_b,                     # 目标相对它的方向
                self._desired_yaw_rate_b,              # 要求它做的自转速度
            ],
            dim=-1,
        )
        return {"policy": obs}

    # 教练打分机制（核心）
    def _get_rewards(self) -> torch.Tensor:
        # 算出目标速度和真实速度的差异。
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_lin_vel_w)
        # torch.linalg.norm 计算直线欧拉距离（误差）。
        lin_vel_error = torch.linalg.norm(desired_lin_vel_b - self._robot.data.root_lin_vel_b, dim=1)
        # 映射函数：1 - tanh(误差)。当误差很大时，tanh趋近1，得分趋近0；误差是0时，tanh是0，拿满分1分。这是一种平滑的打分方式。
        lin_vel_mapped = 1 - torch.tanh(lin_vel_error / 1.0)
        
        # 算出转头速度的误差。
        yaw_rate_error = torch.abs(self._desired_yaw_rate_b.squeeze(-1) - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_mapped = 1 - torch.tanh(yaw_rate_error / 1.0)
        
        # 惩罚项：如果 X轴(滚转) 和 Y轴(俯仰) 旋转过快，就计算平方值作为扣分依据。
        roll_pitch_penalty = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        # 高度考核：要求它尽量稳定在 Z轴 = 1.0 米 的高度。
        z_error = torch.abs(self._robot.data.root_pos_w[:, 2] - 1.0)
        z_pos_mapped = 1 - torch.tanh(z_error / 0.5)

        # 把上面计算的基础分数乘以配置里设定的权重，再乘以 step_dt (每次计算的时间差，保证不同帧率下得分公平)。
        rewards = {
            "tracking_lin_vel": lin_vel_mapped * self.cfg.tracking_lin_vel_reward_scale * self.step_dt,
            "tracking_yaw_rate": yaw_rate_mapped * self.cfg.tracking_yaw_rate_reward_scale * self.step_dt,
            "roll_pitch_penalty": roll_pitch_penalty * self.cfg.roll_pitch_penalty_scale * self.step_dt,
            "z_pos": z_pos_mapped * self.cfg.z_pos_reward_scale * self.step_dt,
        }
        # 把四项得分加总，得出当前帧总分。
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # 记入统计小本本。
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # 判断一局游戏是否结束
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 情况1：时间到了（步数超标）。
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 情况2：坠毁或飞丢了（高度低于 0.1 米 或 高于 2.0 米视为阵亡）。逻辑或运算。
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    # 重置失败的或时间到的无人机环境
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 如果没指名道姓重置哪个，就全部重置。
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # 调用底层重置物理状态。
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # 如果是全局重启，把回合倒计时随机打乱一下，防止所有飞机在同一秒一起重启造成系统卡顿。
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # 动作清零
        self._actions[env_ids] = 0.0
        self._action_history[:, env_ids, :] = 0.0
        
        # --- 下面是一大串复杂的指令重新采样逻辑（给飞机发布新任务） ---
        num_resets = len(env_ids) # 拿到要重置的飞机数量
        rand_probs = torch.rand(num_resets, device=self.device) # 丢骰子决定给它派什么任务
        new_lin_vel = torch.zeros(num_resets, 3, device=self.device)
        
        # 10%-20%概率：只练上下飞 (只有Z轴速度)
        mask_z = (rand_probs >= 0.1) & (rand_probs < 0.2)
        new_lin_vel[mask_z, 2] = torch.empty(mask_z.sum(), device=self.device).uniform_(-1.0, 1.0)
        # 20%-40%概率：只练平飞 (XY轴速度)
        mask_xy = (rand_probs >= 0.2) & (rand_probs < 0.4)
        new_lin_vel[mask_xy, :2] = torch.empty(mask_xy.sum(), 2, device=self.device).uniform_(-1.5, 1.5)
        # 40%以上概率：练习高难度 3D 立体飞行 (XYZ全随机)
        mask_3d = rand_probs >= 0.4
        new_lin_vel[mask_3d, :2] = torch.empty(mask_3d.sum(), 2, device=self.device).uniform_(-1.5, 1.5)
        new_lin_vel[mask_3d, 2] = torch.empty(mask_3d.sum(), device=self.device).uniform_(-0.5, 0.5)
        # 把新任务写入目标本子。
        self._desired_lin_vel_w[env_ids] = new_lin_vel
        
        # 给一定概率派发“旋转机头”的任务
        new_yaw_rate = torch.zeros(num_resets, 1, device=self.device)
        turn_probs = torch.rand(num_resets, device=self.device)
        mask_turn = turn_probs > 0.4
        new_yaw_rate[mask_turn, 0] = torch.empty(mask_turn.sum(), device=self.device).uniform_(-1.5, 1.5)
        self._desired_yaw_rate_b[env_ids] = new_yaw_rate

        # 恢复机器人的默认物理姿势（放回地上对应点，速度清零）
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids] # 定位到各自的坑位
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ==========================================
    # 4. 视觉魔法：在屏幕上画辅助线
    # ==========================================
    # 初始化画笔
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # 如果没创建过画笔，就根据顶部的模板配置依次建立各个图形的实例。
            # 1. 在无人机中心画一个 XYZ 坐标系
            if not hasattr(self, "frame_visualizer"):
                cfg = FRAME_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Robot/CurrentFrame"
                cfg.markers["frame"].scale = (0.1, 0.1, 0.1) 
                self.frame_visualizer = VisualizationMarkers(cfg)

            # 2. 画一个绿色的小球代表：期望速度方向
            if not hasattr(self, "goal_lin_vel_vis"):
                cfg = SPHERE_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/DesiredLinVel"
                cfg.markers["sphere"].radius = 0.02
                # 设置材质为绿色（RGB: 0, 1, 0）
                cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                self.goal_lin_vel_vis = VisualizationMarkers(cfg)

            # 3. 画一个红色小球代表：实际速度方向
            if not hasattr(self, "real_lin_vel_vis"):
                cfg = SPHERE_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/RealLinVel"
                cfg.markers["sphere"].radius = 0.02
                # 设置材质为红色（RGB: 1, 0, 0）
                cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                self.real_lin_vel_vis = VisualizationMarkers(cfg)

            # 4. 青色方块代表期望自转，紫红方块代表实际自转...以此类推
            if not hasattr(self, "goal_ang_vel_vis"):
                cfg = CUBOID_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/DesiredAngVel"
                cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))
                self.goal_ang_vel_vis = VisualizationMarkers(cfg)

            if not hasattr(self, "real_ang_vel_vis"):
                cfg = CUBOID_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/RealAngVel"
                cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
                self.real_ang_vel_vis = VisualizationMarkers(cfg)

            # 统一把它们的“是否可见”属性设为 True（显示出来）
            self.frame_visualizer.set_visibility(True)
            self.goal_lin_vel_vis.set_visibility(True)
            self.real_lin_vel_vis.set_visibility(True)
            self.goal_ang_vel_vis.set_visibility(True)
            self.real_ang_vel_vis.set_visibility(True)
        else:
            # 如果关了调试，就把画笔隐藏掉
            if hasattr(self, "frame_visualizer"): self.frame_visualizer.set_visibility(False)
            if hasattr(self, "goal_lin_vel_vis"): self.goal_lin_vel_vis.set_visibility(False)
            if hasattr(self, "real_lin_vel_vis"): self.real_lin_vel_vis.set_visibility(False)
            if hasattr(self, "goal_ang_vel_vis"): self.goal_ang_vel_vis.set_visibility(False)
            if hasattr(self, "real_ang_vel_vis"): self.real_ang_vel_vis.set_visibility(False)

    # 每一帧渲染时调用的绘图函数（实时更新那些球和方块的位置）
    def _debug_vis_callback(self, event):
        # 还没初始化好就先不画
        if not self._robot.is_initialized:
            return

        # 获取机器人的世界坐标(pos_w)和姿态(quat_w)
        root_pos_w = self._robot.data.root_pos_w 
        root_quat_w = self._robot.data.root_quat_w 

        # 1. 坐标轴直接画在无人机身上
        self.frame_visualizer.visualize(root_pos_w, root_quat_w)

        # 2. 算出一个偏移位置画速度球。不能把球画在中心，不然看不清。速度越大，球离机身越远。
        vel_scale = 0.5  # 乘个0.5当缩小系数
        # 期望位置 = 飞机中心 + 期望速度方向向量 * 0.5
        desired_lin_vel_pos = root_pos_w + self._desired_lin_vel_w * vel_scale
        # 实际位置 = 飞机中心 + 实际速度方向向量 * 0.5
        real_lin_vel_w = self._robot.data.root_lin_vel_w
        real_lin_vel_pos = root_pos_w + real_lin_vel_w * vel_scale
        
        # 指挥底层的画笔在这两个算好的空间位置上把绿球和红球画出来
        self.goal_lin_vel_vis.visualize(desired_lin_vel_pos)
        self.real_lin_vel_vis.visualize(real_lin_vel_pos)

        # 3. 画角速度的方块，逻辑同上，缩小系数为 0.2
        ang_scale = 0.2
        real_ang_vel_w = self._robot.data.root_ang_vel_w
        real_ang_vel_pos = root_pos_w + real_ang_vel_w * ang_scale
        
        # 这里的难点：目标偏航率只设定了Z轴（只管左右转头），我们要用 quat_apply（四元数正向旋转）把它映射回世界坐标里，否则在外面看方向就不准了。
        desired_ang_vel_b = torch.zeros_like(real_ang_vel_w)
        desired_ang_vel_b[:, 2] = self._desired_yaw_rate_b[:, 0]
        desired_ang_vel_w = quat_apply(root_quat_w, desired_ang_vel_b)
        desired_ang_vel_pos = root_pos_w + desired_ang_vel_w * ang_scale

        # 画出方块
        self.goal_ang_vel_vis.visualize(desired_ang_vel_pos)
        self.real_ang_vel_vis.visualize(real_ang_vel_pos)