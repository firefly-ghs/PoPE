import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from PoPE_io import PoPEIO

class PoPE(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        # 预计算频率 theta (与 RoPE 相同)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 可学习的偏置 delta_c，初始化为 0 或均匀分布
        # 对应 d/2 个复数分量
        self.delta = nn.Parameter(torch.zeros(head_dim // 2))
        
        # 论文建议将 delta 限制在 [-2pi, 0] 之间
        #self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        # 形状: (seq_len, head_dim/2)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, q, k, seq_len):
        # q, k 形状: (batch, seq_len, heads, head_dim)
        
        # 1. 将向量两两分组，计算模长 mu
        # q_split: (batch, seq_len, heads, head_dim/2, 2)
        ## GPT-J style
        q_split = q.reshape(*q.shape[:-1], -1, 2)
        k_split = k.reshape(*k.shape[:-1], -1, 2)
        # 计算模长: mu = sqrt(x^2 + y^2)
        # 论文提到可以使用 softplus 或 ReLU 进一步处理模长，这里采用标准二范数
        mu_q = torch.norm(q_split, p=2, dim=-1) # (batch, seq_len, q_heads, head_dim/2) 
        mu_k = torch.norm(k_split, p=2, dim=-1) # (batch, seq_len, k_heads, head_dim/2) 
        
        # 2. 获取预计算的 cos 和 sin (对应位置 t 和 s)
        cos = self.cos_cached[:seq_len, :] # (seq_len, dim/2)
        sin = self.sin_cached[:seq_len, :]
        
        # 3. 构造 PoPE 转换后的 Q 和 K
        # 对于 Query: q_new = mu_q * exp(i * t * theta)
        # 将 cos/sin 扩展为 (1, seq_len, 1, dim/2) 以匹配 mu_q 的形状 (batch, seq_len, heads, dim/2)
        cos_r = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim/2)
        sin_r = sin.unsqueeze(0).unsqueeze(2)
        q_real = mu_q * cos_r
        q_imag = mu_q * sin_r
        
        # 对于 Key: k_new = mu_k * exp(i * (s * theta + delta))
        # 使用复数乘法公式: exp(i(A+B)) = exp(iA)exp(iB)
        # = (cosA cosB - sinA sinB) + i(sinA cosB + cosA sinB)
        delta_cos = torch.cos(self.delta)
        delta_sin = torch.sin(self.delta)
        
        k_cos_total = cos * delta_cos - sin * delta_sin
        k_sin_total = sin * delta_cos + cos * delta_sin
        
        k_cos_r = k_cos_total.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim/2)
        k_sin_r = k_sin_total.unsqueeze(0).unsqueeze(2)
        k_real = mu_k * k_cos_r
        k_imag = mu_k * k_sin_r
        
        # 4. 重新组合成笛卡尔坐标形式输出
        # 将 real 和 imag 拼接回原来的维度
        q_pope = torch.stack([q_real, q_imag], dim=-1).flatten(-2)
        k_pope = torch.stack([k_real, k_imag], dim=-1).flatten(-2)
        
        return q_pope, k_pope

if __name__ == "__main__":
    # 参数设置: 
    batch, seq_len, q_heads, k_heads, head_dim = 1, 256, 128, 1, 64
    
    # 1. 初始化IO管理器
    data_dir = "data"
    pope_io = PoPEIO(base_dir=data_dir, device="cpu")
   
    # 2. 生成输入数据
    q, k = pope_io.generate_input_data(batch, seq_len, q_heads, k_heads, head_dim)
    print(f"输入数据形状: Q={q.shape}, K={k.shape}")
    
    # 3. 使用PoPE层
    pope_layer = PoPE(head_dim=head_dim)
    q_out, k_out = pope_layer(q, k, seq_len)
    
    # 4. 保存结果
    pope_io.save_pope_results(q_out, k_out, "q_py_output.bin", "k_py_output.bin")
    print(f"输出数据已保存到: {data_dir}")
