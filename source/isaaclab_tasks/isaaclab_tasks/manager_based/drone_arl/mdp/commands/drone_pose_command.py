
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# 版权所有 (c) 2022-2026, Isaac Lab 项目开发者。保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX 许可证标识符：BSD-3-Clause

"""Sub-module containing command generators for pose tracking.
包含用于位姿跟踪的指令生成器的子模块。
"""

from __future__ import annotations # 允许在类型提示中使用未定义的类名

import torch # 引入 PyTorch 进行张量批处理计算

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand # 引入通用位姿指令基类
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error # 引入坐标系变换和误差计算的数学工具


class DroneUniformPoseCommand(UniformPoseCommand):
    """Drone-specific UniformPoseCommand extensions.
    针对无人机特有的 UniformPoseCommand 扩展。

    This class customizes the generic :class:`UniformPoseCommand` for drone (multirotor)
    use-cases. Main differences and additions:
    此类为无人机（多旋翼飞行器）用例定制了通用的 `UniformPoseCommand`。主要区别和新增内容包括：

    - Transforms pose commands from the drone's base frame to the world frame before use.
      在使用前，将位姿指令从无人机的基座坐标系转换到全局世界坐标系。
    - Accounts for per-environment origin offsets (``scene.env_origins``) when computing
      position errors so tasks running on shifted/sub-terrain environments compute
      meaningful errors.
      在计算位置误差时，考虑了每个环境的原点偏移量（`scene.env_origins`），以便在偏移/子地形环境上运行的任务能够计算出有意义的误差（防止所有并行环境的指令堆叠在原点）。
    - Computes and exposes simple metrics used by higher-level code: ``position_error``
      and ``orientation_error`` (stored in ``self.metrics``).
      计算并暴露供高层（强化学习）代码使用的简单指标：`position_error`（位置误差）和 `orientation_error`（姿态误差）（存储在 `self.metrics` 字典中）。
    - Provides a debug visualization callback that renders the goal pose (with
      sub-terrain shift) and current body pose using the existing visualizers.
      提供了一个调试可视化回调函数，使用现有的可视化器在仿真 GUI 中渲染目标位姿（带有子地形偏移）和当前机身位姿。

    The implementation overrides :meth:`_update_metrics` and :meth:`_debug_vis_callback`
    from the base class to implement these drone-specific behaviors.
    该实现重写了基类中的 `_update_metrics` 和 `_debug_vis_callback` 方法，以实现这些针对无人机的特定行为。
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        # 翻译：将指令从基座坐标系转换到仿真世界坐标系
        # 结构分析：[:, :3] 提取三维坐标位置，[:, 3:] 提取四元数旋转。将基于世界坐标系的根节点绝对位姿与基于基座的相对指令进行组合。
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,  # 机器人在世界坐标系下的根节点位置
            self.robot.data.root_quat_w, # 机器人在世界坐标系下的根节点姿态(四元数)
            self.pose_command_b[:, :3],  # 指令在基座局部坐标系下的相对位置
            self.pose_command_b[:, 3:],  # 指令在基座局部坐标系下的相对姿态
        )
        
        # compute the error
        # 翻译：计算误差（调用 compute_pose_error 分别返回位置误差张量和旋转误差张量）
        pos_error, rot_error = compute_pose_error(
            # Sub-terrain shift for correct position error calculation @grzemal
            # 翻译与分析：为计算正确的位置误差进行子地形偏移 @grzemal (代码作者备注)
            # 核心原理：必须加上 env_origins (各个并行局部环境在全局世界中的偏移量) 才能得到正确的绝对目标位置基准
            self.pose_command_b[:, :3] + self._env.scene.env_origins, 
            self.pose_command_w[:, 3:], # 目标在世界坐标系下的绝对旋转
            self.robot.data.body_pos_w[:, self.body_idx],  # 无人机机身特定刚体在世界坐标系下的当前位置
            self.robot.data.body_quat_w[:, self.body_idx], # 无人机机身特定刚体在世界坐标系下的当前旋转
        )
        
        # 翻译与分析：计算位置和姿态的欧几里得距离（L2范数 dim=-1 计算最后一个维度的模长），
        # 存入 metrics 字典，供强化学习的奖励函数（Reward Function）计算惩罚项调用
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        # 翻译与分析：检查机器人是否已初始化。注意：这是安全检查机制，为了防止机器人在仿真重置时被取消初始化，导致尝试访问其底层 data 属性而引发程序崩溃。
        if not self.robot.is_initialized:
            return
            
        # update the markers
        # 翻译：更新可视化标记
        
        # -- goal pose
        # 翻译：-- 目标位姿
        # Sub-terrain shift for visualization purposes @grzemal
        # 翻译：为可视化目的进行子地形偏移 @grzemal。在图形界面渲染时同样需要加上环境偏移量，否则所有并行环境的目标都会画在世界原点。
        self.goal_pose_visualizer.visualize(
            self.pose_command_b[:, :3] + self._env.scene.env_origins, self.pose_command_b[:, 3:]
        )
        
        # -- current body pose
        # 翻译：-- 当前机身位姿
        # 分析：获取 7 维位姿张量，利用张量切片 [:, :3] 分离位置送入第一参数，[:, 3:7] 分离四元数送入第二参数进行渲染。
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])