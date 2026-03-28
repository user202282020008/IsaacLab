# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
# from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse
# === 替换数学导入 ===
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse, quat_apply

# === 确保引入了所有的可视化标记 ===
from isaaclab.markers import FRAME_MARKER_CFG, SPHERE_MARKER_CFG, CUBOID_MARKER_CFG


from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
# 新增：导入标准坐标系和球体标记配置
from isaaclab.markers import FRAME_MARKER_CFG, SPHERE_MARKER_CFG # isort: skip



















class QuadcopterEnvWindow(BaseEnvWindow):
    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # 在 GUI 调试窗口注册 Targets 可视化开关
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 13 
    state_space = 0
    # 关键字：开启调试可视化，自动启用 _debug_vis_callback
    debug_vis = True 

    ui_window_class_type = QuadcopterEnvWindow

    # Sim2Real 核心参数
    action_delay_steps = 2         # 延迟步数 (40ms)
    lin_vel_noise_std = 0.1        # 线速度噪声
    ang_vel_noise_std = 0.2        # 角速度噪声

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,

    physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,                 # <--- 修正后
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # 奖励权重
    tracking_lin_vel_reward_scale = 10.0
    tracking_yaw_rate_reward_scale = 5.0
    roll_pitch_penalty_scale = -0.05
    z_pos_reward_scale = 1.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Sim2Real: 动作延迟缓冲区 [延迟步数, 环境数, 动作维度]
        self._action_history = torch.zeros(self.cfg.action_delay_steps + 1, self.num_envs, 4, device=self.device)
        
        # 目标指令张量 [Nenvs, XYZ]
        self._desired_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device) 
        # [Nenvs, YawRate标量]
        self._desired_yaw_rate_b = torch.zeros(self.num_envs, 1, device=self.device) 

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["tracking_lin_vel", "tracking_yaw_rate", "roll_pitch_penalty", "z_pos"]
        }
        
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # 初始化可视化系统 (DirectRLEnv 底层会自动调用 _set_debug_vis_impl)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # 延迟逻辑不变
        clamped_actions = actions.clone().clamp(-1.0, 1.0)
        self._action_history = torch.roll(self._action_history, shifts=1, dims=0)
        self._action_history[0] = clamped_actions
        delayed_actions = self._action_history[-1]
        self._actions = delayed_actions
        
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        # 噪声观测逻辑不变
        real_lin_vel_b = self._robot.data.root_lin_vel_b
        real_ang_vel_b = self._robot.data.root_ang_vel_b
        obs_lin_vel_b = real_lin_vel_b + torch.randn_like(real_lin_vel_b) * self.cfg.lin_vel_noise_std
        obs_ang_vel_b = real_ang_vel_b + torch.randn_like(real_ang_vel_b) * self.cfg.ang_vel_noise_std
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_lin_vel_w)
        
        obs = torch.cat(
            [
                obs_lin_vel_b,
                obs_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_lin_vel_b,
                self._desired_yaw_rate_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 奖励计算逻辑不变
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_lin_vel_w)
        lin_vel_error = torch.linalg.norm(desired_lin_vel_b - self._robot.data.root_lin_vel_b, dim=1)
        lin_vel_mapped = 1 - torch.tanh(lin_vel_error / 1.0)
        yaw_rate_error = torch.abs(self._desired_yaw_rate_b.squeeze(-1) - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_mapped = 1 - torch.tanh(yaw_rate_error / 1.0)
        roll_pitch_penalty = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        z_error = torch.abs(self._robot.data.root_pos_w[:, 2] - 1.0)
        z_pos_mapped = 1 - torch.tanh(z_error / 0.5)

        rewards = {
            "tracking_lin_vel": lin_vel_mapped * self.cfg.tracking_lin_vel_reward_scale * self.step_dt,
            "tracking_yaw_rate": yaw_rate_mapped * self.cfg.tracking_yaw_rate_reward_scale * self.step_dt,
            "roll_pitch_penalty": roll_pitch_penalty * self.cfg.roll_pitch_penalty_scale * self.step_dt,
            "z_pos": z_pos_mapped * self.cfg.z_pos_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._action_history[:, env_ids, :] = 0.0
        
        # 复杂混合指令采样逻辑不变
        num_resets = len(env_ids)
        rand_probs = torch.rand(num_resets, device=self.device)
        new_lin_vel = torch.zeros(num_resets, 3, device=self.device)
        mask_z = (rand_probs >= 0.1) & (rand_probs < 0.2)
        new_lin_vel[mask_z, 2] = torch.empty(mask_z.sum(), device=self.device).uniform_(-1.0, 1.0)
        mask_xy = (rand_probs >= 0.2) & (rand_probs < 0.4)
        new_lin_vel[mask_xy, :2] = torch.empty(mask_xy.sum(), 2, device=self.device).uniform_(-1.5, 1.5)
        mask_3d = rand_probs >= 0.4
        new_lin_vel[mask_3d, :2] = torch.empty(mask_3d.sum(), 2, device=self.device).uniform_(-1.5, 1.5)
        new_lin_vel[mask_3d, 2] = torch.empty(mask_3d.sum(), device=self.device).uniform_(-0.5, 0.5)
        self._desired_lin_vel_w[env_ids] = new_lin_vel
        
        new_yaw_rate = torch.zeros(num_resets, 1, device=self.device)
        turn_probs = torch.rand(num_resets, device=self.device)
        mask_turn = turn_probs > 0.4
        new_yaw_rate[mask_turn, 0] = torch.empty(mask_turn.sum(), device=self.device).uniform_(-1.5, 1.5)
        self._desired_yaw_rate_b[env_ids] = new_yaw_rate

        # 重置物理状态数据不变
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # === 修改 可视化实现入口 (Tokens level analysis) ===

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # 1. 机身坐标系 (显示朝向)
            if not hasattr(self, "frame_visualizer"):
                cfg = FRAME_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Robot/CurrentFrame"
                cfg.markers["frame"].scale = (0.1, 0.1, 0.1) 
                self.frame_visualizer = VisualizationMarkers(cfg)

            # 2. 🟢 期望线速度 (绿色球)
            if not hasattr(self, "goal_lin_vel_vis"):
                cfg = SPHERE_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/DesiredLinVel"
                cfg.markers["sphere"].radius = 0.02
                # 【修复材质报错】使用 visual_material 和 PreviewSurfaceCfg
                cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                self.goal_lin_vel_vis = VisualizationMarkers(cfg)

            # 3. 🔴 实际物理线速度 (红色球)
            if not hasattr(self, "real_lin_vel_vis"):
                cfg = SPHERE_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/RealLinVel"
                cfg.markers["sphere"].radius = 0.02
                cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                self.real_lin_vel_vis = VisualizationMarkers(cfg)

            # 4. 🔵 期望角速度 (青色小方块)
            if not hasattr(self, "goal_ang_vel_vis"):
                cfg = CUBOID_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/DesiredAngVel"
                cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))
                self.goal_ang_vel_vis = VisualizationMarkers(cfg)

            # 5. 🟣 实际物理角速度 (紫红色小方块)
            if not hasattr(self, "real_ang_vel_vis"):
                cfg = CUBOID_MARKER_CFG.copy() 
                cfg.prim_path = "/Visuals/Command/RealAngVel"
                cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
                self.real_ang_vel_vis = VisualizationMarkers(cfg)

            # 统一开启可见性
            self.frame_visualizer.set_visibility(True)
            self.goal_lin_vel_vis.set_visibility(True)
            self.real_lin_vel_vis.set_visibility(True)
            self.goal_ang_vel_vis.set_visibility(True)
            self.real_ang_vel_vis.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"): self.frame_visualizer.set_visibility(False)
            if hasattr(self, "goal_lin_vel_vis"): self.goal_lin_vel_vis.set_visibility(False)
            if hasattr(self, "real_lin_vel_vis"): self.real_lin_vel_vis.set_visibility(False)
            if hasattr(self, "goal_ang_vel_vis"): self.goal_ang_vel_vis.set_visibility(False)
            if hasattr(self, "real_ang_vel_vis"): self.real_ang_vel_vis.set_visibility(False)






            
    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        root_pos_w = self._robot.data.root_pos_w 
        root_quat_w = self._robot.data.root_quat_w 

        # 1. 渲染坐标轴
        self.frame_visualizer.visualize(root_pos_w, root_quat_w)

        # 2. 渲染线速度 (乘以 0.5 缩放因子，让球离机身有一定距离以指示方向)
        vel_scale = 0.5
        # 目标线速度位置 (绿球)
        desired_lin_vel_pos = root_pos_w + self._desired_lin_vel_w * vel_scale
        # 实际线速度位置 (红球)
        real_lin_vel_w = self._robot.data.root_lin_vel_w
        real_lin_vel_pos = root_pos_w + real_lin_vel_w * vel_scale
        
        self.goal_lin_vel_vis.visualize(desired_lin_vel_pos)
        self.real_lin_vel_vis.visualize(real_lin_vel_pos)

        # 3. 渲染角速度 (乘以 0.2 缩放因子)
        ang_scale = 0.2
        # 实际角速度世界坐标 (紫红方块)
        real_ang_vel_w = self._robot.data.root_ang_vel_w
        real_ang_vel_pos = root_pos_w + real_ang_vel_w * ang_scale
        
        # 目标角速度: 只有局部 Z 轴的 Yaw Rate。需要通过 quat_apply 旋转到世界坐标系 (青色方块)
        desired_ang_vel_b = torch.zeros_like(real_ang_vel_w)
        desired_ang_vel_b[:, 2] = self._desired_yaw_rate_b[:, 0]
        desired_ang_vel_w = quat_apply(root_quat_w, desired_ang_vel_b)
        desired_ang_vel_pos = root_pos_w + desired_ang_vel_w * ang_scale

        self.goal_ang_vel_vis.visualize(desired_ang_vel_pos)
        self.real_ang_vel_vis.visualize(real_ang_vel_pos)