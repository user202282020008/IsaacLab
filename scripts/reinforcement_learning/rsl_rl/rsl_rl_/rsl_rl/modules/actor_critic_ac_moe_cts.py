# -*- coding: utf-8 -*-
'''
@File    : actor_critic_moe_cts.py
@Desc    : Multiplicative Compositional Policies Concurrent Teacher Student Network
'''
import numpy as np

# 导入 PyTorch 核心模块
import torch  # 已修正原代码中的乱码
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 导入外部定义的强化学习组件（MLP基础网络、MoE架构、归一化层等）
from rsl_rl.modules.utils import get_activation, MLP, MoE, Experts, L2Norm, SimNorm

class ActorCriticACMoECTS(nn.Module):
    # 静态属性，标识该网络是否包含 RNN/LSTM 等循环结构。
    # 这里设为 False，因为历史信息是通过固定长度的张量拼接（History Buffer）处理的，而非隐藏状态传递。
    is_recurrent = False 

    def __init__(self,  
                 num_obs,  # 每帧基础观测的维度（如关节角度、角速度、机身姿态等）
                 num_critic_obs,  # Critic 的观测维度，通常等于基础观测 + 特权信息（如摩擦力、地形高度图）
                 num_actions,  # 网络输出的动作维度（如12个电机的目标力矩或位置）
                 num_envs,  # 并行训练的环境总数（用于初始化 history 缓冲区的 batch size）
                 history_length,  # 堆叠的历史观测帧数（学生网络通过历史推断特权信息）
                 actor_hidden_dims=[512, 256, 128],  # Actor 内各专家 MLP 的隐藏层节点数
                 critic_hidden_dims=[512, 256, 128],  # Critic 内各专家 MLP 的隐藏层节点数
                 teacher_encoder_hidden_dims=[512, 256],  # 教师编码器（处理特权信息）的 MLP 结构
                 student_encoder_hidden_dims=[512, 256],  # 学生编码器（处理历史观测）的 MLP 结构
                 expert_num=8,  # MoE 架构中的专家数量 $N$
                 activation='elu',  # 激活函数，ELU 在强化学习中常用于平滑梯度
                 init_noise_std=1.0,  # 探索噪声的初始标准差（控制初始探索范围）
                 latent_dim=32,  # 教师/学生编码器输出的隐向量维度（浓缩环境特征的特征空间）
                 norm_type='l2norm',  # 隐向量的归一化方式，防止表示空间崩溃
                 **kwargs):  # 吸收额外未使用的超参数
        
        # 容错处理：打印并忽略传入的未知参数，保持代码健壮性
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        # 强制断言：确保归一化类型是已实现的两种之一。
        # 归一化在此处至关重要，它确保了学生输出的 latent 和教师输出的 latent 在同一个度量空间内，便于计算蒸馏损失（Loss）。
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        
        super().__init__()
        
        # --- 实例属性挂载 ---
        # 将局部变量保存为对象的全局状态，供整个生命周期内的 forward、act、evaluate 等方法使用
        self.num_actions = num_actions
        self.history_length = history_length

        # --- 维度计算 ---
        # 教师编码器输入：全部特权信息
        mlp_input_dim_t = num_critic_obs  
        # 学生编码器输入：历史帧数 × 单帧观测维度（将时间序列展平为一维向量处理）
        mlp_input_dim_s = num_obs * history_length  
        # Critic 输入：隐向量（提取出的环境特征） + 特权信息（确保 Critic 具备完美价值评估能力）
        mlp_input_dim_c = latent_dim + num_critic_obs  
        # Actor 输入：隐向量 + 当前基础观测（Actor 只能根据当前状态和特征环境行事）
        mlp_input_dim_a = latent_dim + num_obs  

        # --- 注册历史缓冲区 ---
        # persistent=False 表示这个 buffer 虽然随模型保存在 GPU 上，但不作为网络权重保存到 state_dict 中（因为它是环境状态，不是模型参数）
        # 形状：[环境数量, 历史帧数, 观测维度]
        self.register_buffer("history", torch.zeros((num_envs, history_length, num_obs)), persistent=False)

        # --- 网络模块实例化 ---
        
        # 1. 教师编码器：将高维特权信息压缩到 latent_dim 维度的隐空间，并归一化
        self.teacher_encoder = nn.Sequential(
            MLP([mlp_input_dim_t, *teacher_encoder_hidden_dims, latent_dim], activation),
            L2Norm() if norm_type == 'l2norm' else SimNorm()
        )

        # 2. 学生编码器：尝试仅用历史观测信息，预测出与教师相同的隐向量
        self.student_encoder = nn.Sequential(
            MLP([mlp_input_dim_s, *student_encoder_hidden_dims, latent_dim], activation),
            L2Norm() if norm_type == 'l2norm' else SimNorm()
        )

        # 3. Actor MoE (演员混合专家)：
        # 根据特征(latent)和观测(obs)，门控网络分配权重，各专家输出动作，最后加权求和得到动作均值
        self.actor_moe = MoE(
            expert_num=expert_num,
            input_dim=mlp_input_dim_a,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

        # 4. Critic Experts (评论家专家群)：
        # 注意这里使用的是 Experts 而非完整的 MoE。这是因为 Critic 直接复用 Actor 算出的专家权重，
        # 以保证 Actor 和 Critic 在策略路由上的高度一致性（解决同一类地形使用同一套评估逻辑）。
        self.critic_experts = Experts(
            expert_num=expert_num,
            input_dim=mlp_input_dim_c,
            backbone_hidden_dims=critic_hidden_dims[:-1],
            expert_hidden_dim=critic_hidden_dims[-1],
            output_dim=1, # 每个专家输出一个标量价值 (Value)
            activation=activation,
        )

        # 打印网络结构，方便调试
        print(f"Actor MoE: {self.actor_moe}")
        print(f"Critic Experts: {self.critic_experts}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student Encoder: {self.student_encoder}")

        # --- 动作分布设置 ---
        self.distribution = None
        # 将标准差设为可学习的参数 (nn.Parameter)，初始值为传入的 init_noise_std
        # 强化学习通过这个标准差控制探索（Exploration），随着训练进行，标准差通常会逐渐减小
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))  
        # 关闭 Normal 分布的输入校验。在成千上万个并行环境中频繁采样时，关闭校验可显著提升训练 FPS (帧率)
        Normal.set_default_validate_args = False

    # 重置被截断/终止环境的历史观测
    def reset(self, dones=None):
        # 利用布尔索引，将死亡/重置环境的历史清零，避免跨越 episode 的信息污染
        self.history[dones > 0] = 0.0

    def forward(self):
        # 强制子类或外部调用特定的 act/evaluate 方法，禁止直接调用模型本身的前向传播
        raise NotImplementedError
    
    # 装饰器，将方法转为只读属性，方便直接读取分布的均值（动作期望）
    @property
    def action_mean(self):
        return self.distribution.mean

    # 方便直接读取分布的标准差（探索力度）
    @property
    def action_std(self):
        return self.distribution.stddev
    
    # 计算当前策略的香农熵，用于 PPO 算法中的熵奖励（Entropy Bonus），鼓励探索
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # 核心动作分布更新逻辑
    def update_distribution(self, x):
        # 传入 x (隐向量+观测)，通过 Actor MoE 输出动作均值
        mean, _ = self.actor_moe(x)
        # 构造正态分布：均值为专家网络的输出，标准差为独立的可学习参数
        # mean*0. 是一种 PyTorch 技巧，用于将可学习的标量/向量参数 broadcast(广播) 到与 mean 相同的 batch shape 上
        self.distribution = Normal(mean, mean*0. + self.std)

    # 训练时的动作采样逻辑
    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        # 核心：CTS 并发架构的路由分发
        if is_teacher:
            # 教师模式：直接使用特权信息提取精准的 latent
            latent = self.teacher_encoder(privileged_obs)
        else:
            # 学生模式：利用无梯度的学生网络（通常在训练初期不产生主导梯度）从历史中提取 latent
            with torch.no_grad():
                latent = self.student_encoder(history)
        
        # 将提纯的环境特征与当前机械物理状态拼接
        x = torch.cat([latent, obs], dim=1)
        self.update_distribution(x)
        # 从构建的正态分布中采样出带噪声的动作，供环境执行
        return self.distribution.sample()
    
    # 计算已采取动作在当前分布下的对数概率，用于 PPO 算法计算 Importance Ratio (重要性采样比)
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 部署到现实机器人 (Deployment/Inference) 时使用的方法
    def act_inference(self, obs):
        # 维护历史队列：丢弃最旧的一帧 ([:, 1:])，在末尾拼接最新的一帧 obs
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        # 将三维张量 (Batch, Length, Dim) 展平为二维矩阵 (Batch, Length*Dim)，送入学生编码器
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        # 推理时不需要探索，直接取分布的均值（确定性策略 Deterministic Policy）
        mean, weights = self.actor_moe(x)
        return mean

    # 价值评估，计算优势函数 (Advantage) 时使用
    def evaluate(self, obs, privileged_obs, history, is_teacher, **kwargs):
        # 同样的 CTS 路由获取 latent
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history)
            
        x_actor = torch.cat([latent, obs], dim=1)
        # 关键设计：通过 Actor 的门控网络 (gating_network) 计算出不同专家的置信度权重 (B, expert_num)
        weights = self.actor_moe.gating_network(x_actor)  
        
        # Critic 输入拼接：注意 latent.detach()！
        # 截断梯度回传，防止 Critic 的价值损失影响编码器的表征学习，确保 latent 仅由动作策略或监督损失塑造
        x_critic = torch.cat([latent.detach(), privileged_obs], dim=1)
        # 并行计算所有 Critic 专家的价值
        experts_value = self.critic_experts(x_critic)
        
        # 使用 Actor 算出的权重，对 Critic 专家的价值进行加权求和，得到最终的状态价值 V(s)
        value = torch.sum(weights.unsqueeze(-1) * experts_value, dim=1)
        return value, weights