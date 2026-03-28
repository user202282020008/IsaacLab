import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel

class MoEActor(MLPModel):
    # 注意这里的方法签名，这是新版 rsl_rl 的标准接口
    def __init__(self, obs, obs_groups, name, action_dim, **kwargs):
        # 1. 调用基类初始化，让 rsl_rl 帮你处理复杂的 obs 解析和归一化
        super().__init__(obs, obs_groups, name, action_dim, **kwargs)
        
        # 2. 获取处理后的输入维度
        input_dim = self._get_latent_dim()
        
        # 3. 彻底覆盖默认的 self.mlp！在这里写入你自己的网络结构
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim) # Quadcopter 的 action_dim 通常是 4
        )
        print("====== Custom MoE Actor Initialized Successfully! ======")