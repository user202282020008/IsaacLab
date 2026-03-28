# -*- coding: utf-8 -*-
'''
@File    : actor_critic_cts.py
@Time    : 2025/12/30 21:06:08
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Concurrent Teacher Student Network
@Refer   : CTS https://arxiv.org/abs/2405.10830
'''
import numpy as np

# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 并发教师-学生结构的Actor-Critic网络
class ActorCriticCTS(nn.Module):
    is_recurrent = False  # 是否为循环网络，这里为False
    def __init__(self,  num_actor_obs,  # actor输入观测维度
                        num_critic_obs,  # critic输入观测维度
                        num_actions,  # 动作空间维度
                        num_envs,  # 并行环境数
                        history_length,  # 历史观测长度
                        actor_hidden_dims=[512, 256, 128],  # actor隐藏层结构
                        critic_hidden_dims=[512, 256, 128],  # critic隐藏层结构
                        teacher_encoder_hidden_dims=[512, 256],  # 教师编码器隐藏层
                        student_encoder_hidden_dims=[512, 256],  # 学生编码器隐藏层
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

        activation = get_activation(activation)  # 获取激活函数实例

        # 计算各模块输入维度
        mlp_input_dim_t = num_critic_obs  # 教师编码器输入
        mlp_input_dim_s = num_actor_obs * history_length  # 学生编码器输入
        mlp_input_dim_a = latent_dim + num_actor_obs  # actor输入
        mlp_input_dim_c = latent_dim + num_critic_obs  # critic输入

        # 历史观测缓存，形状(环境数, 历史长度, 观测维度)
        self.history = torch.zeros((num_envs, history_length, num_actor_obs), device='cuda')

        # 构建教师编码器MLP+归一化
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_t, teacher_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(teacher_encoder_hidden_dims)):
            if l == len(teacher_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], latent_dim))
                if norm_type == 'l2norm':
                    encoder_layers.append(L2Norm())
                elif norm_type == 'simnorm':
                    encoder_layers.append(SimNorm())
            else:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], teacher_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.teacher_encoder = nn.Sequential(*encoder_layers)

        # 构建学生编码器MLP+归一化
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_s, student_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(student_encoder_hidden_dims)):
            if l == len(student_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], latent_dim))
                if norm_type == 'l2norm':
                    encoder_layers.append(L2Norm())
                elif norm_type == 'simnorm':
                    encoder_layers.append(SimNorm())
            else:
                encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], student_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.student_encoder = nn.Sequential(*encoder_layers)

        # 构建actor策略网络MLP
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # 构建critic价值网络MLP
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # 打印各子网络结构
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Teacher Encoder MLP: {self.teacher_encoder}")
        print(f"Student Encoder MLP: {self.student_encoder}")

        # 动作噪声参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # 关闭Normal分布的参数校验以加速
        Normal.set_default_validate_args = False
        
        # 预留权重初始化（未启用）
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # 线性层权重初始化（当前未用）
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    # 重置历史观测
    def reset(self, dones=None):
        self.history[dones > 0] = 0.0

    # 前向传播未实现
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
    def update_distribution(self, latent_and_obs):
        mean = self.actor(latent_and_obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    # 采样动作
    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history).detach()
        x = torch.cat([latent, obs], dim=1)
        self.update_distribution(x)
        return self.distribution.sample()
    
    # 计算动作对数概率
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 推理时采样动作均值
    def act_inference(self, obs):
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        actions_mean = self.actor(x)
        return actions_mean

    # 评估价值
    def evaluate(self, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history)
        x = torch.cat([latent.detach(), privileged_obs], dim=1)
        value = self.critic(x)
        return value

# 激活函数工厂，根据字符串返回对应激活层
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

# L2范数归一化层
class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)

# Simplicial归一化层
class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """
    def __init__(self):
        super().__init__()
        self.dim = 8  # for latent dim 512
    def forward(self, x):
        shp = x.shape
        target_shape = list(shp[:0]) + [-1, self.dim]
        x = x.view(target_shape)
        x = F.softmax(x, dim=0)
        return x.view(shp)
    def __repr__(self):
        return f"SimNorm(dim={self.dim})"
