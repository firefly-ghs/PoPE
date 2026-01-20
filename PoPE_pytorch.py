import torch
import torch.nn as nn
import torch.nn.functional as F

class PoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        # 预计算频率 theta (与 RoPE 相同)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 可学习的偏置 delta_c，初始化为 0 或均匀分布
        # 对应 d/2 个复数分量
        self.delta = nn.Parameter(torch.zeros(dim // 2))
        
        # 论文建议将 delta 限制在 [-2pi, 0] 之间
        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        # 形状: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, q, k, seq_len):
        # q, k 形状: (batch, num_heads, seq_len, head_dim)
        
        # 1. 将向量两两分组，计算模长 mu
        # q_split: (batch, heads, seq_len, head_dim/2, 2)
        q_split = q.reshape(*q.shape[:-1], -1, 2)
        k_split = k.reshape(*k.shape[:-1], -1, 2)
        
        # 计算模长: mu = sqrt(x^2 + y^2)
        # 论文提到可以使用 softplus 或 ReLU 进一步处理模长，这里采用标准二范数
        mu_q = torch.norm(q_split, p=2, dim=-1) # (batch, heads, seq_len, dim/2)
        mu_k = torch.norm(k_split, p=2, dim=-1)
        
        # 2. 获取预计算的 cos 和 sin (对应位置 t 和 s)
        cos = self.cos_cached[:seq_len, :] # (seq_len, dim/2)
        sin = self.sin_cached[:seq_len, :]
        
        # 3. 构造 PoPE 转换后的 Q 和 K
        # 对于 Query: q_new = mu_q * exp(i * t * theta)
        q_real = mu_q * cos.unsqueeze(0).unsqueeze(0)
        q_imag = mu_q * sin.unsqueeze(0).unsqueeze(0)
        
        # 对于 Key: k_new = mu_k * exp(i * (s * theta + delta))
        # 使用复数乘法公式: exp(i(A+B)) = exp(iA)exp(iB)
        # = (cosA cosB - sinA sinB) + i(sinA cosB + cosA sinB)
        delta_cos = torch.cos(self.delta)
        delta_sin = torch.sin(self.delta)
        
        k_cos_total = cos * delta_cos - sin * delta_sin
        k_sin_total = sin * delta_cos + cos * delta_sin
        
        k_real = mu_k * k_cos_total.unsqueeze(0).unsqueeze(0)
        k_imag = mu_k * k_sin_total.unsqueeze(0).unsqueeze(0)
        
        # 4. 重新组合成笛卡尔坐标形式输出
        # 将 real 和 imag 拼接回原来的维度
        q_pope = torch.stack([q_real, q_imag], dim=-1).flatten(-2)
        k_pope = torch.stack([k_real, k_imag], dim=-1).flatten(-2)
        
        return q_pope, k_pope

# 示例使用
batch, heads, seq, d_head = 2, 8, 128, 64
q = torch.randn(batch, heads, seq, d_head)
k = torch.randn(batch, heads, seq, d_head)

pope_layer = PoPE(d_head)
q_out, k_out = pope_layer(q, k, seq)

print(f"Output shape: {q_out.shape}") # 应为 (2, 8, 128, 64)
