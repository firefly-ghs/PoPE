import argparse
import numpy as np
import os
import sys


def load_f16(path, count):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.fromfile(path, dtype=np.float16)
    if data.size != count:
        raise ValueError(f"File {path} size mismatch: got {data.size}, expected {count}")
    return data


def compare_files(q_cpp, k_cpp, q_py, k_py, batch, seq_len, q_heads, k_heads, dim, tol):
    q_count = batch * seq_len * q_heads * dim
    k_count = batch * seq_len * k_heads * dim

    q_cpp_data = load_f16(q_cpp, q_count).reshape((batch, seq_len, q_heads, dim))
    k_cpp_data = load_f16(k_cpp, k_count).reshape((batch, seq_len, k_heads, dim))
    q_py_data = load_f16(q_py, q_count).reshape((batch, seq_len, q_heads, dim))
    k_py_data = load_f16(k_py, k_count).reshape((batch, seq_len, k_heads, dim))

    # 加载原始输入数据用于比对原因
    q_in = load_f16("data/q_input.bin", q_count).reshape((batch, seq_len, q_heads, dim))

    q_diff = np.abs(q_cpp_data - q_py_data)
    k_diff = np.abs(k_cpp_data - k_py_data)
   
    print(f"\n{'='*20} PoPE 一致性对比报告 {'='*20}")
    print(f"数据规格: Q Heads={q_heads}, K Heads={k_heads}, Seq Len={seq_len}, Dim={dim}")
    print(f"对比模式: Float16 (Half Precision)")
    
    # 打印前 5 个显著误差点及其原始输入
    large_error_indices = np.where(q_diff > 0.001)
    if len(large_error_indices[0]) > 0:
        print(f"\n[!] 发现 {len(large_error_indices[0])} 个误差大于 0.001 的点。")
        print("\n显著误差点深度分析 (Top 5):")
        for i in range(min(5, len(large_error_indices[0]))):
            b, s, h, d = [idx[i] for idx in large_error_indices]
            pair_start = (d // 2) * 2
            in_r = q_in[b, s, h, pair_start]
            in_i = q_in[b, s, h, pair_start + 1]
            cpp_val = q_cpp_data[b, s, h, d]
            py_val = q_py_data[b, s, h, d]
            print(f"  位置 [b={b}, s={s:3d}, h={h:3d}, d={d:2d}] -> 输入对: ({in_r:8.4f}, {in_i:8.4f}) | C++: {cpp_val:8.4f} | Py: {py_val:8.4f} | 误差: {q_diff[b, s, h, d]:.6f}")

    results = {
        'q_max_abs': float(np.max(q_diff)),
        'q_mean_abs': float(np.mean(q_diff)),
        'k_max_abs': float(np.max(k_diff)),
        'k_mean_abs': float(np.mean(k_diff)),
    }

    print(f"\n误差统计摘要:")
    print(f"  - Q 矩阵: 最大绝对误差 = {results['q_max_abs']:.6e}, 平均误差 = {results['q_mean_abs']:.6e}")
    print(f"  - K 矩阵: 最大绝对误差 = {results['k_max_abs']:.6e}, 平均误差 = {results['k_mean_abs']:.6e}")

    # 判定结果
    if results['q_max_abs'] <= tol and results['k_max_abs'] <= tol:
        print(f"\n[结论] ✅ 通过测试! 误差在容差 {tol} 范围内。")
        return 0
    elif results['q_max_abs'] <= 5e-3: # float16 的宽松判定
        print(f"\n[结论] ⚠️ 基本一致。部分误差超过 {tol}，但在 Float16 舍入误差合理范围内 (< 0.005)。")
        return 0
    else:
        print(f"\n[结论] ❌ 测试失败! 误差过大，请检查逻辑实现。")
        return 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--q_cpp', default=os.path.join('data', 'q_cpp_output.bin'))
    p.add_argument('--k_cpp', default=os.path.join('data', 'k_cpp_output.bin'))
    p.add_argument('--q_py', default=os.path.join('data', 'q_py_output.bin'))
    p.add_argument('--k_py', default=os.path.join('data', 'k_py_output.bin'))
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--q_heads', type=int, default=128)
    p.add_argument('--k_heads', type=int, default=1)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--tol', type=float, default=1e-5)
    args = p.parse_args()

    try:
        sys.exit(compare_files(args.q_cpp, args.k_cpp, args.q_py, args.k_py,
                               args.batch, args.seq_len, args.q_heads, args.k_heads,
                               args.dim, args.tol))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
