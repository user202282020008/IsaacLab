import os
import mujoco
import mujoco.viewer
import pygame
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 0. 定义与 RSL-RL 1:1 匹配的极简飞控大脑
# ==========================================
class MinimalActor(nn.Module):
    def __init__(self):
        super().__init__()
        # 13 维观测输入，4 维动作输出。默认隐藏层为 [256, 128, 64]
        self.actor = nn.Sequential(
            nn.Linear(13, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.actor(x)

# ==========================================
# 1. 手柄初始化 (Pygame)
# ==========================================
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("❌ 错误：没有检测到手柄！")
    exit()
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"🎮 成功连接手柄: {joystick.get_name()}")

# ==========================================
# 2. 加载 MuJoCo 与 PyTorch 权重
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "quadcopter.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

device = torch.device("cpu")
policy = MinimalActor().to(device)

# 动态加载 policy.pt 权重
policy_path = os.path.join(script_dir, "policy.pt")
if not os.path.exists(policy_path):
    print(f"❌ 找不到权重文件！请把训练好的模型重命名为 'policy.pt' 并放到: {script_dir}")
    exit()



# 只提取 Actor (策略网络) 的权重，智能兼容不同的 RSL-RL 存档格式
checkpoint = torch.load(policy_path, map_location=device)

# 判断是否套了 'model_state_dict' 外壳
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint # 直接就是权重字典本身

# 提取以 'actor.' 开头的神经元权重
actor_state_dict = {k: v for k, v in state_dict.items() if k.startswith('actor.')}

# 防翻车检查：如果提取出来是空的，把里面的钥匙打印出来看看
if len(actor_state_dict) == 0:
    print("⚠️ 警告：没找到 'actor.' 权重！这个文件里究竟有什么？\n", list(state_dict.keys()))
else:
    policy.load_state_dict(actor_state_dict)


policy.eval() # 开启推理模式
print("🧠 AI 飞控大脑加载完毕！")

# ==========================================
# 3. 核心观测函数 (Isaac Lab 数学复刻)
# ==========================================
def get_mujoco_obs(data, desired_lin_vel_w, desired_yaw_rate_b):
    quat_wxyz = data.qpos[3:7].copy()
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) 
    
    real_lin_vel_w = data.qvel[:3].copy()
    real_lin_vel_b = rot.inv().apply(real_lin_vel_w)
    real_ang_vel_b = data.qvel[3:6].copy()
    
    gravity_w = np.array([0.0, 0.0, -1.0])
    projected_gravity_b = rot.inv().apply(gravity_w)
    
    # 核心：将世界系的手柄目标速度，转换到当前的机身局部坐标系
    desired_lin_vel_b = rot.inv().apply(desired_lin_vel_w)
    
    obs_np = np.concatenate([
        real_lin_vel_b,          
        real_ang_vel_b,          
        projected_gravity_b,     
        desired_lin_vel_b,       
        [desired_yaw_rate_b]     
    ])
    return torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)

def apply_deadzone(value, threshold=0.1):
    """手柄摇杆死区，防止摇杆松动导致漂移"""
    return 0.0 if abs(value) < threshold else value

# ==========================================
# 4. 主物理闭环控制 (带 Debug 监控)
# ==========================================
control_dt = 0.02 
physics_steps_per_control = int(control_dt / model.opt.timestep)

print("起飞前准备就绪！请使用手柄摇杆控制无人机，按下窗口关闭按钮退出。")
print("提示：左摇杆控制水平移动，右摇杆控制升降和偏航。")


# 增加一个计数器，防止打印刷屏太快看不清
print_counter = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pygame.event.pump()
        
        # --- A. 读取手柄 ---
        # 我们暂时把乘数缩小，防止乱飞
        cmd_x = -apply_deadzone(joystick.get_axis(1)) * 1.0   
        cmd_y = -apply_deadzone(joystick.get_axis(0)) * 1.0   
        
        # ⚠️ 注意这里：Xbox 在 Linux 下右摇杆 Y 轴可能是 3 或 4，X 轴可能是 2 或 3
        # 如果手柄扳机键（LT/RT）被误触，也会占用轴编号！
        cmd_z = -apply_deadzone(joystick.get_axis(3)) * 1.0   
        cmd_yaw = -apply_deadzone(joystick.get_axis(2)) * 1.0 
        
        desired_lin_vel_w = np.array([cmd_x, cmd_y, cmd_z])
        desired_yaw_rate_b = cmd_yaw
        
        # --- B. 组装观测并推理 ---
        obs_tensor = get_mujoco_obs(data, desired_lin_vel_w, desired_yaw_rate_b)
        with torch.no_grad():
            action_tensor = policy(obs_tensor)
        
        actions = action_tensor.squeeze(0).cpu().numpy()
        actions = np.clip(actions, -1.0, 1.0)
        
        # --- 🔍 Debug 脑电波打印 (每秒打印 2 次) ---
        print_counter += 1
        if print_counter % 25 == 0:
            print("\n" + "="*40)
            print(f"🎮 手柄目标指令 | X速:{cmd_x:5.2f}  Y速:{cmd_y:5.2f}  Z速(升降):{cmd_z:5.2f}  Yaw速:{cmd_yaw:5.2f}")
            print(f"🧠 AI 原始输出   | 推力:{actions[0]:5.2f}  滚转:{actions[1]:5.2f}  俯仰:{actions[2]:5.2f}  偏航:{actions[3]:5.2f}")
        
# --- C. 动作映射到 4 个独立电机 (Motor Thrusts) ---
        
        # 1. 把 AI 输出的 [-1, 1] 解码为油门比例 [0, 1]
        motor_u = (actions + 1.0) / 2.0 
        
        # 2. 计算 Crazyflie 极限推力
        # 飞机重 0.027kg，重力为 0.265N。推重比约 1.9，总推力极限约 0.5N
        # 平摊到 4 个电机，每个电机的极限推力约为 0.125 N
        max_thrust_per_motor = 0.125 
        
        # 3. 计算出 4 个电机当前的真实推力 (单位：牛顿)
        thrusts = motor_u * max_thrust_per_motor
        
        # 4. 直接把推力写入 MuJoCo 的 4 个电机里！
        # (因为这 4 个力不在重心，它们一高一低，自然就产生了滚转和俯仰！)
        data.ctrl[0] = thrusts[0] # 左前电机 (FL)
        data.ctrl[1] = thrusts[1] # 左后电机 (BL)
        data.ctrl[2] = thrusts[2] # 右后电机 (BR)
        data.ctrl[3] = thrusts[3] # 右前电机 (FR)