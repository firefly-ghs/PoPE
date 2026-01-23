# pope_io.py
import os
import numpy as np
import torch

class PoPEIO:
    """PoPE数据的统一IO管理类"""
    
    def __init__(self, base_dir=".", device="cpu"):
        """
        初始化IO管理器
        :param base_dir: 数据保存/读取的基础目录
        :param device: 数据加载后的目标设备（cpu/cuda）
        """
        self.base_dir = base_dir
        self.device = device
        # 确保目录存在
        os.makedirs(self.base_dir, exist_ok=True)
    
    def get_file_path(self, filename):
        """生成完整的文件路径（处理Windows/Linux路径兼容）"""
        return os.path.join(self.base_dir, filename)
    
    def generate_input_data(self, batch, seq_len, q_heads, k_heads, head_dim, 
                           dtype=torch.float16, save=True):
        """生成PoPE输入数据"""
        q = torch.randn(batch, seq_len, q_heads, head_dim, dtype=dtype, device=self.device)
        k = torch.randn(batch, seq_len, k_heads, head_dim, dtype=dtype, device=self.device)
        
        if save:
            self.save_tensor(q, "q_input.bin")
            self.save_tensor(k, "k_input.bin")
            print(f"PoPE输入数据已保存到{self.base_dir}")
        return q, k
    
    def save_tensor(self, tensor, filename, as_f16=True):
        """保存张量到二进制文件（自动处理设备转换）。

        浮点张量默认会以 float16（二进制 IEEE-754 binary16）格式写入磁盘，
        以便与 C++ 端的 f16 二进制读取兼容。非浮点张量保留原始 dtype。
        设置 `as_f16=False` 可绕过此行为并以原始 dtype 保存。
        """
        # 确保数据在CPU上再保存
        # 如果张量需要梯度，先 detach
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        arr = tensor.numpy()
        # 对于浮点类型，默认转换为 float16 保存
        if as_f16 and arr.dtype.kind == 'f':
            arr = arr.astype(np.float16)
        arr.tofile(self.get_file_path(filename))
    
    def load_tensor(self, filename, shape, dtype=np.float16):
        """从二进制文件加载张量（自动转换到目标设备）"""
        file_path = self.get_file_path(filename)
        data = np.fromfile(file_path, dtype=dtype)
        tensor = torch.from_numpy(data.reshape(shape))
        return tensor.to(self.device)
    
    def save_pope_results(self, q_out, k_out, q_name, k_name):
        """保存PoPE处理结果"""
        self.save_tensor(q_out, q_name)
        self.save_tensor(k_out, k_name)
    
    def load_input_data(self, batch, seq_len, q_heads, k_heads, head_dim):
        """加载PoPE输入数据"""
        q_shape = (batch, seq_len, q_heads, head_dim)
        k_shape = (batch, seq_len, k_heads, head_dim)
        q = self.load_tensor("data/q_input.bin", q_shape)
        k = self.load_tensor("data/k_input.bin", k_shape)
        return q, k
    
    # 未来HIP代码的扩展接口预留
    def load_hip_results(self, batch, seq_len, heads, head_dim):
        """加载HIP实现的结果（预留接口）"""
        raise NotImplementedError("HIP结果加载功能待实现")
    
    def compare_results(self, pytorch_results, hip_results, tolerance=1e-6):
        """比较不同实现的结果一致性（预留接口）"""
        q_diff = torch.abs(pytorch_results[0] - hip_results[0]).max()
        k_diff = torch.abs(pytorch_results[1] - hip_results[1]).max()
        print(f"Q差异: {q_diff.item()}, K差异: {k_diff.item()}")
        return q_diff < tolerance and k_diff < tolerance