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

    # 添加打印q部分数据结果的代码
    print("\n=== Q数据对比（前几个元素）===")
    # 打印C++实现的q数据的前几个元素
    print("C++ Q数据（batch=0, seq=0, head=0, dim=0-7）:")
    print(q_cpp_data[0, 0, 0, :8])
    print("\nPython Q数据（batch=0, seq=0, head=0, dim=0-7）:")
    print(q_py_data[0, 0, 0, :8])
    print("\nQ数据差异（batch=0, seq=0, head=0, dim=0-7）:")
    print(np.abs(q_cpp_data[0, 0, 0, :8] - q_py_data[0, 0, 0, :8]))
    
    # 打印更多数据用于比较
    print("\nC++ Q数据（batch=0, seq=1, head=0, dim=0-7）:")
    print(q_cpp_data[0, 1, 0, :8])
    print("\nPython Q数据（batch=0, seq=1, head=0, dim=0-7）:")
    print(q_py_data[0, 1, 0, :8])
    print("\nQ数据差异（batch=0, seq=1, head=0, dim=0-7）:")
    print(np.abs(q_cpp_data[0, 1, 0, :8] - q_py_data[0, 1, 0, :8]))

    q_diff = np.abs(q_cpp_data - q_py_data)
    k_diff = np.abs(k_cpp_data - k_py_data)
   
    # 添加打印误差较大的数据及其位置
    large_error_threshold = tol * 1000  # 定义较大误差的阈值，这里使用10倍的容差
    max_print_count = 5  # 每个矩阵最多打印5个较大误差点
    
    print("\n=== 误差较大的数据点（Q矩阵）===")
    q_large_error_indices = np.where(q_diff > large_error_threshold)
    q_large_error_count = len(q_large_error_indices[0])
    print(f"Q矩阵中误差超过{large_error_threshold:.6e}的点共有: {q_large_error_count}个")
    
    if q_large_error_count > 0:
        print(f"前{min(max_print_count, q_large_error_count)}个较大误差点:")
        for i in range(min(max_print_count, q_large_error_count)):
            b_idx = q_large_error_indices[0][i]
            s_idx = q_large_error_indices[1][i]
            h_idx = q_large_error_indices[2][i]
            d_idx = q_large_error_indices[3][i]
            
            cpp_val = q_cpp_data[b_idx, s_idx, h_idx, d_idx]
            py_val = q_py_data[b_idx, s_idx, h_idx, d_idx]
            diff_val = q_diff[b_idx, s_idx, h_idx, d_idx]
            
            print(f"位置: (batch={b_idx}, seq={s_idx}, head={h_idx}, dim={d_idx}) - C++: {cpp_val:.6f}, Python: {py_val:.6f}, 误差: {diff_val:.6f}")
    
    results = {
        'q_max_abs': float(np.max(q_diff)),
        'q_mean_abs': float(np.mean(q_diff)),
        'k_max_abs': float(np.max(k_diff)),
        'k_mean_abs': float(np.mean(k_diff)),
    }

    print("Comparison results:")
    print(f" Q max_abs={results['q_max_abs']:.6e}, mean_abs={results['q_mean_abs']:.6e}")
    print(f" K max_abs={results['k_max_abs']:.6e}, mean_abs={results['k_mean_abs']:.6e}")

    ok = (results['q_max_abs'] <= tol) and (results['k_max_abs'] <= tol)
    if ok:
        print("PASS: within tolerance")
        return 0
    else:
        print(f"FAIL: max abs exceeds tol={tol}")
        return 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--q_cpp', default='q_cpp_output.bin')
    p.add_argument('--k_cpp', default='k_cpp_output.bin')
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
        rc = compare_files(args.q_cpp, args.k_cpp, args.q_py, args.k_py,
                           args.batch, args.seq_len, args.q_heads, args.k_heads, args.dim, args.tol)
        sys.exit(rc)
    except Exception as e:
        print("Error during compare:", e)
        sys.exit(3)


if __name__ == '__main__':
    main()
