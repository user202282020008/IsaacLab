# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from abc import ABC, abstractmethod # 用于定义抽象基类和抽象方法
import torch
from typing import Tuple, Union

# minimal interface of the environment
class VecEnv(ABC):
    """向量化环境的抽象基类，定义了强化学习环境中必需的接口和属性。
    
    该类提供了与多个并行环境交互的标准接口，支持批量操作以提高训练效率。
    所有具体的环境实现都必须继承此类并实现其抽象方法。
    """
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor 
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor # current episode duration
    extras: dict
    device: torch.device
    
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        """执行环境中的一步操作。
        
        Args:
            actions: 形状为[num_envs, num_actions]的动作张量，包含每个环境要执行的动作
            
        Returns:
            Tuple包含以下元素:
            - obs: 形状为[num_envs, num_obs]的观测张量
            - privileged_obs: 形状为[num_envs, num_privileged_obs]的特权观测张量，如果没有则为None
            - rewards: 形状为[num_envs]的奖励张量
            - dones: 形状为[num_envs]的完成标志张量，表示每个环境是否结束
            - infos: 包含额外信息的字典
        """
        pass
        
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        """重置指定的环境实例。
        
        Args:
            env_ids: 要重置的环境ID列表或张量，可以是Python列表或torch.Tensor
        """
        pass
        
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        """获取当前所有环境的观测值。
        
        Returns:
            形状为[num_envs, num_obs]的观测张量
        """
        pass
        
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        """获取当前所有环境的特权观测值。
        
        特权观测通常包含在训练期间可用但在部署时不可用的额外信息。
        
        Returns:
            形状为[num_envs, num_privileged_obs]的特权观测张量，如果没有特权观测则返回None
        """
        pass