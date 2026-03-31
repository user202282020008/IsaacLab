import os
import mujoco
import mujoco.viewer
import pygame
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 0. 决策层：定义与 RSL-RL 1:1 匹配的极简飞控大脑
# ==========================================
class MinimalActor(nn.Module):
    def __init__(self):
        super().__init__()
        # 13 维观测输入，4 维动作输出。默认隐藏层为 [256, 128, 64]
        # Sequential 是 PyTorch 的流水线容器，数据会按顺序穿过这些网络层。
        self.actor = nn.Sequential(
            nn.Linear(13, 256),
            nn.ELU(),  # ELU 是一种激活函数，给网络加入非线性思考能力
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 4)
        )

    # forward 定义了数据流向，每次调用神经网络时都会执行这里
    def forward(self, x):
        return self.actor(x)

# ==========================================
# 1. 指令层：手柄初始化 (Pygame)
# ==========================================
# 初始化 pygame 引擎
pygame.init()
# 初始化游戏手柄模块
pygame.joystick.init()

# 安全检查：如果没有插手柄，立刻停止程序
if pygame.joystick.get_count() == 0:
    print("❌ 错误：没有检测到手柄！")
    exit()

# 获取系统的第一个手柄（编号 0）并激活
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"🎮 成功连接手柄: {joystick.get_name()}")

# 手柄摇杆死区函数：如果摇杆只被轻轻碰了一下（值小于 threshold），就当作 0 处理，防止飞机自己乱漂
def apply_deadzone(value, threshold=0.002):
    return 0.0 if abs(value) < threshold else value

# ==========================================
# 2. 感知层：核心观测函数 (数学复刻)
# ==========================================
# 这个函数的作用是把MuJoCo的世界坐标数据，翻译成AI习惯的机身相对坐标系数据
def get_mujoco_obs(data, desired_lin_vel_w, desired_yaw_rate_b):
    # 获取四元数姿态 [w, x, y, z] 并转为 scipy 库需要的 [x, y, z, w] 格式
    quat_wxyz = data.qpos[3:7].copy()
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) 
    
    # rot 旋转姿态

    # 获取世界系下的线速度，并用旋转矩阵的逆 (rot.inv) 转换到机体局部坐标系
    real_lin_vel_w = data.qvel[:3].copy()
    real_lin_vel_b = rot.inv().apply(real_lin_vel_w)
    
    # 获取机体局部的角速度
    real_ang_vel_b = data.qvel[3:6].copy()
    
    # 计算重力向量在无人机当前倾斜姿态下的投影（让 AI 知道自己目前歪了多少）
    gravity_w = np.array([0.0, 0.0, -1.0])
    projected_gravity_b = rot.inv().apply(gravity_w)
    
    # 核心：将手柄下达的世界系目标速度，转换到当前的机身局部坐标系
    desired_lin_vel_b = rot.inv().apply(desired_lin_vel_w)
    
    # 把所有数据拼成一根长长的数据条 (13个数字)
    obs_np = np.concatenate([
        real_lin_vel_b,          
        real_ang_vel_b,          
        projected_gravity_b,     
        desired_lin_vel_b,       
        [desired_yaw_rate_b]     
    ])
    # 转换为 PyTorch 需要的张量，增加一个 Batch 维度 (unsqueeze(0))，并传送到计算设备
    return torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)


# ==========================================
# 3. 初始化加载：MuJoCo 与 PyTorch 权重
# ==========================================
# 获取当前 Python 脚本所在的文件夹路径
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "quadcopter.xml")

# 将 XML 图纸加载为 MuJoCo 的静态模型 (model) 和动态数据容器 (data)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

device = torch.device("cpu")
policy = MinimalActor().to(device)

# 动态加载 policy.pt 权重
policy_path = os.path.join(script_dir, "policy.pt")
if not os.path.exists(policy_path):
    print(f"❌ 找不到权重文件！请把训练好的模型重命名为 'policy.pt' 并放到: {script_dir}")
    exit()

# 提取 Actor 权重，兼容不同的存档格式
checkpoint = torch.load(policy_path, map_location=device)
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
actor_state_dict = {k: v for k, v in state_dict.items() if k.startswith('actor.')}

if len(actor_state_dict) == 0:
    print("⚠️ 警告：没找到 'actor.' 权重！\n", list(state_dict.keys()))
else:
    policy.load_state_dict(actor_state_dict)

# 开启推理模式（关闭 Dropout 等训练特有的随机干扰机制）
policy.eval() 
print("🧠 AI 飞控大脑加载完毕！")

# ==========================================
# 4. 执行与物理层：主物理闭环控制
# ==========================================
# 控制频率设置：AI 大脑每 0.02 秒做一次决策（50Hz）
control_dt = 0.02 
# 物理引擎底层算得很快，算一下 AI 每做一次决定，物理引擎需要跑多少个小步长
physics_steps_per_control = int(control_dt / model.opt.timestep)

print("起飞前准备就绪！请使用手柄摇杆控制无人机，按下窗口关闭按钮退出。")
print_counter = 0

# 启动 MuJoCo 的被动渲染窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 只要窗口没被关掉，就一直循环运行
    while viewer.is_running():
        # 刷新系统事件，确保手柄输入被系统接收
        pygame.event.pump()
        
# --- A. 读取手柄指令 (已解绑并放大灵敏度系数) ---
        
        # 定义系统的物理极限边界 (根据 AI 训练时的 uniform_ 范围设定)
        MAX_LIN_VEL_XY = 2.0  # 水平极限速度：2.0 m/s (比训练时的 1.5 略大，压榨极限)
        MAX_LIN_VEL_Z = 1.0   # 升降极限速度：1.0 m/s
        MAX_YAW_RATE = 3.0    # 偏航极限角速度：3.0 rad/s (约 170度/秒，转头更灵活)
        
        # 读取摇杆值，应用死区后，乘以对应的极限物理系数
        # 注意：这里的负号取决于你手柄的具体物理朝向，如果发现前后反了，把对应的负号去掉即可
        cmd_x = apply_deadzone(joystick.get_axis(1)) * MAX_LIN_VEL_XY   
        cmd_y = apply_deadzone(joystick.get_axis(0)) * MAX_LIN_VEL_XY   
        cmd_z = -apply_deadzone(joystick.get_axis(3)) * MAX_LIN_VEL_Z   
        cmd_yaw = -apply_deadzone(joystick.get_axis(2)) * MAX_YAW_RATE 
        
        desired_lin_vel_w = np.array([cmd_x, cmd_y, cmd_z])
        desired_yaw_rate_b = cmd_yaw
        
        # --- B. 感知与决策推理 ---
        obs_tensor = get_mujoco_obs(data, desired_lin_vel_w, desired_yaw_rate_b)
        
        # with torch.no_grad() 告诉 PyTorch 现在是实战，不需要记录数学倒数（梯度），节省内存并提速
        with torch.no_grad():
            action_tensor = policy(obs_tensor)
        
        # 把神经网络输出的张量转换为普通的 NumPy 数组，并限制在 [-1.0, 1.0] 范围内防止异常暴走
        actions_o = action_tensor.squeeze(0).cpu().numpy()
        actions = np.clip(actions_o, -1.0, 1.0)

        
        # --- C. 分配层：混控矩阵 (Control Allocation) ---
        robot_mass = 0.027 
        gravity = 9.81
        robot_weight = robot_mass * gravity

        # 1. 把 AI 的无量纲输出，还原为带有物理单位的总推力 Fz 和 力矩 Mx, My, Mz
        Fz = 1.9 * robot_weight * (actions[0] + 1.0) / 2.0
        Mx = 0.1 * actions[1]
        My = 0.1 * actions[2]
        Mz = 0.1 * actions[3]
        target_wrench = np.array([Fz, Mx, My, Mz])

        # 2. 构建控制分配矩阵 A (按 FL, BL, BR, FR 顺序设定)
        L = 0.046 / np.sqrt(2) # 假设对角线为 0.046 米
        C_tau = 0.015          # 偏航反扭矩系数
        A = np.array([
            [1.0,    1.0,    1.0,    1.0],      # 总推力 Z
            [L,      L,     -L,     -L],        # 滚转 X
            [-L,     L,      L,     -L],        # 俯仰 Y
            [C_tau, -C_tau,  C_tau, -C_tau]     # 偏航 Z
        ])

        # 3. 矩阵求伪逆，解算出四个电机各自应该出多少力
        A_inv = np.linalg.pinv(A)
        motor_thrusts = A_inv @ target_wrench

        # # 4. 电机不能倒转往下拉，所以下限为 0；上限设为 0.125N
        max_thrust_per_motor = 0.125
        motor_thrusts = np.clip(motor_thrusts, -max_thrust_per_motor, max_thrust_per_motor)

        # 5. 写入 MuJoCo 底层电机的致动器 (Actuators)
        data.ctrl[0] = motor_thrusts[0] # 左前电机 (FL)
        data.ctrl[1] = motor_thrusts[1] # 左后电机 (BL)
        data.ctrl[2] = motor_thrusts[2] # 右后电机 (BR)
        data.ctrl[3] = motor_thrusts[3] # 右前电机 (FR)

        # --- D. 推进物理时间线与画面同步 (极其关键) ---
        # 推动物理引擎的内部时间向前走
        for _ in range(physics_steps_per_control):
            mujoco.mj_step(model, data)
            
        # 物理状态计算完毕，锁住渲染器并更新屏幕画面
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.sync()

        # --- E. 脑电波打印 (每秒打印 2 次，不干扰运行) ---
        print_counter += 1
        if print_counter % 25 == 0:
    # --- 终极无闪烁全屏仪表盘 ---
            
            # 1. 组装面板内容
            dashboard = (
                f"{'='*50}\n"
                f"🎯 目标指令 | X:{cmd_x:5.2f}  Y:{cmd_y:5.2f}  Z:{cmd_z:5.2f}  Yaw:{cmd_yaw:5.2f}\n"
                f"🧠 AI 输出  | 推:{actions[0]:5.2f}  滚:{actions[1]:5.2f}  俯:{actions[2]:5.2f}  偏:{actions[3]:5.2f}\n"
                f"⚙️ 电机推力 | FL:{motor_thrusts[0]:5.3f} BL:{motor_thrusts[1]:5.3f} BR:{motor_thrusts[2]:5.3f} FR:{motor_thrusts[3]:5.3f}\n"
                f"🤖 原始输出: [{actions_o[0]:5.2f}, {actions_o[1]:5.2f}, {actions_o[2]:5.2f}, {actions_o[3]:5.2f}]\n"
                f"{'='*50}"
            )

            # 2. 绝对定位刷新
            # \033[H  -> 回到屏幕绝对左上角
            # \033[0J -> 清除屏幕上的旧残骸
            # end=""  -> 防止多余的换行导致画面跳动
            print(f"\033[H\033[0J{dashboard}", end="", flush=True)