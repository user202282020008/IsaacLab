# -*- coding: utf-8 -*-
'''
@File    : actor_critic_moe_cts.py
@Time    : 2025/12/30 21:06:46
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Multiplicative Compositional Policies Concurrent Teacher Student Network
@Refer   :  CTS https://arxiv.org/abs/2405.10830,
            Switch Transformers (Load Balance) https://arxiv.org/abs/2101.03961
            MoE-Loco (AC MoE) http://arxiv.org/abs/2503.08564
'''
import numpy as np

# 导入PyTorch相关模块
import torch
import torch.nn as nn
from torch.distributions import Normal
# 导入自定义工具模块，包括MLP、MoE、学生MoE编码器、专家网络、归一化等
from rsl_rl.modules.utils import MLP, MoE, StudentMoEEncoder, Experts, L2Norm, SimNorm

# 双MoE结构的并发教师-学生Actor-Critic网络
class ActorCriticDualMoECTS(nn.Module):
    is_recurrent = False  # 是否为循环网络，这里为False
    def __init__(self,  num_obs,  # 观测维度
                        num_critic_obs,  # 评论者观测维度
                        num_actions,  # 动作空间维度
                        num_envs,  # 并行环境数
                        history_length,  # 历史观测长度
                        actor_hidden_dims=[512, 256, 128],  # actor隐藏层结构
                        critic_hidden_dims=[512, 256, 128],  # critic隐藏层结构
                        teacher_encoder_hidden_dims=[512, 256],  # 教师编码器隐藏层
                        student_encoder_hidden_dims=[512, 256, 256],  # 学生MoE编码器隐藏层，最后一层为专家隐藏层
                        expert_num=8,  # 专家数量
                        activation='elu',  # 激活函数类型
                        init_noise_std=1.0,  # 动作噪声初始标准差
                        latent_dim=32,  # 编码器输出潜在空间维度
                        norm_type='l2norm',  # 归一化类型
                        **kwargs):  # 其他参数
        # 处理多余参数
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        # 只支持l2norm和simnorm两种归一化
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super().__init__()
        self.num_actions = num_actions
        self.history_length = history_length

        # 计算各模块输入维度
        mlp_input_dim_t = num_critic_obs  # 教师编码器输入
        mlp_input_dim_s = num_obs * history_length  # 学生编码器输入
        mlp_input_dim_c = latent_dim + num_critic_obs  # 评论者输入
        mlp_input_dim_a = latent_dim + num_obs  # actor输入

        # 历史观测缓存，形状(环境数, 历史长度, 观测维度)
        self.register_buffer("history", torch.zeros((num_envs, history_length, num_obs)), persistent=False)

        # 教师编码器：MLP+归一化
        self.teacher_encoder = nn.Sequential(
            MLP([mlp_input_dim_t, *teacher_encoder_hidden_dims, latent_dim], activation),
            L2Norm() if norm_type == 'l2norm' else SimNorm()
        )

        # 学生MoE编码器
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=mlp_input_dim_s,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )

        # MCP Actor：MoE结构的策略网络
        self.actor_moe = MoE(
            expert_num=expert_num,
            input_dim=mlp_input_dim_a,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

        # 评论者（Critic）：多专家结构
        self.critic_experts = Experts(
            expert_num=expert_num,
            input_dim=mlp_input_dim_c,
            backbone_hidden_dims=critic_hidden_dims[:-1],
            expert_hidden_dim=critic_hidden_dims[-1],
            output_dim=1,
            activation=activation,
        )

        # 打印各子模块结构
        print(f"Actor MoE: {self.actor_moe}")
        print(f"Critic Experts: {self.critic_experts}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student MoE Encoder: {self.student_moe_encoder}")

        # 动作分布相关参数
        self.distribution = None
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))  # 动作噪声参数
        # 关闭Normal分布的参数校验以加速
        Normal.set_default_validate_args = False

    # 重置历史观测（如环境重置时）
    def reset(self, dones=None):
        self.history[dones > 0] = 0.0

    # 前向传播未实现，需子类实现
    def forward(self):
        raise NotImplementedError
    
    # 动作均值属性
    @property
    def action_mean(self):
        return self.distribution.mean

    # 动作标准差属性
    @property
    def action_std(self):
        return self.distribution.stddev
    
    # 策略熵属性
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # 更新动作分布
    def update_distribution(self, x):
        mean, _ = self.actor_moe(x)
        self.distribution = Normal(mean, mean*0. + self.std)

    # 采样动作
    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            with torch.no_grad():
                latent, _ = self.student_moe_encoder(history)
        x = torch.cat([latent, obs], dim=1)
        self.update_distribution(x)
        return self.distribution.sample()
    
    # 计算动作对数概率
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 推理时采样动作均值
    def act_inference(self, obs):
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent, _ = self.student_moe_encoder(self.history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        mean, _ = self.actor_moe(x)
        return mean

    # 评估价值和专家权重
    def evaluate(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent, _ = self.student_moe_encoder(history)
        x_actor = torch.cat([latent, obs], dim=1)
        weights = self.actor_moe.gating_network(x_actor)  # (B, expert_num)
        x_critic = torch.cat([latent.detach(), privileged_obs], dim=1)
        experts_value = self.critic_experts(x_critic)
        value = torch.sum(weights.unsqueeze(-1) * experts_value, dim=1)
        return value, weights
