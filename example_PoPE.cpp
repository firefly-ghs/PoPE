#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>

// 配置参数 (需与 Python 脚本一致)
const int batch = 1;
const int seq_len = 256;
const int q_heads = 128; 
const int k_heads = 1;   
const int dim = 64;

void load_bin(const std::string& path, float* data, size_t size) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) { std::cerr << "无法打开文件: " << path << std::endl; 
    std::filesystem::path current_path = std::filesystem::current_path();
    std::cout << "Warning: Current Path: " << current_path.string() << std::endl; exit(1); }
    std::streamsize actual = is.tellg();
    std::streamsize expected = static_cast<std::streamsize>(size) * static_cast<std::streamsize>(sizeof(float));
    if (actual != expected) {
        std::cerr << "警告: 文件大小不匹配: " << path << ". 期望 " << expected << " 字节, 实际 " << actual << " 字节\n";
    }
    is.seekg(0);
    is.read(reinterpret_cast<char*>(data), expected);
    if (!is) { std::cerr << "读取错误: " << path << std::endl; exit(1); }
}

void save_bin(const std::string& path, float* data, size_t size) {
    std::ofstream os(path, std::ios::binary);
    os.write(reinterpret_cast<char*>(data), size * sizeof(float));
}

int main() {
    // --- 1. 调用 Python 生成输入（使用 workspace 中的 pytorch_PoPE_2.py） ---
    std::cout << "Step 1: 正在调用 Python 生成原始矩阵..." << std::endl;
    int ret1 = std::system("python d:\\Projects\\13-PoPE\\0-code\\pytorch_PoPE_2.py");
    if (ret1 != 0) { std::cerr << "调用 Python 生成数据失败, 返回码=" << ret1 << std::endl; return -1; }

    // --- 2. 加载数据 ---
    // 使用与 Python 一致的输入文件名: q_input.bin / k_input.bin
    std::vector<float> q_in(batch * seq_len * q_heads * dim), k_in(batch * seq_len * k_heads * dim);
    std::vector<float> delta(dim / 2, 0.0f), inv_freq(dim / 2);
    load_bin("q_input.bin", q_in.data(), q_in.size());
    load_bin("k_input.bin", k_in.data(), k_in.size());

    // Python 端通常没有单独保存 inv_freq/delta，这里按 PoPE 的公式重建 in    g++ -O2 -std=c++17 d:\Projects\13-PoPE\0-code\example_PoPE.cpp -o d:\Projects\13-PoPE\0-code\example_PoPE.exev_freq，delta 默认为 0
    const float base = 10000.0f;
    for (int i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(base, static_cast<float>((2 * i)) / static_cast<float>(dim));
        // delta[i] 保持为 0 （与 PyTorch 初始化一致）
    }

    // --- 3. C++ 执行 PoPE 逻辑 ---
    std::cout << "Step 2: C++ 正在执行 PoPE 计算..." << std::endl;
    std::vector<float> q_cpp(batch * seq_len * q_heads * dim), k_cpp(batch * seq_len * k_heads * dim);

    for (int b = 0; b < batch; ++b) {
        for (int l = 0; l < seq_len; ++l) { // Position
            // 处理 Query
            for (int hq = 0; hq < q_heads; ++hq) {
                for (int d_idx = 0; d_idx < dim / 2; ++d_idx) {
                    int idx_real_q = ((b * seq_len + l) * q_heads + hq) * dim + (d_idx * 2);
                    int idx_imag_q = idx_real_q + 1;
                    float q_r = q_in[idx_real_q];
                    float q_i = q_in[idx_imag_q];
                    float mu_q = std::sqrt(q_r * q_r + q_i * q_i);
                    float theta = static_cast<float>(l) * inv_freq[d_idx];
                    q_cpp[idx_real_q] = mu_q * std::cos(theta);
                    q_cpp[idx_imag_q] = mu_q * std::sin(theta);
                }
            }

            // 处理 Key（HK）
            for (int hk = 0; hk < k_heads; ++hk) {
                for (int d_idx = 0; d_idx < dim / 2; ++d_idx) {
                    int idx_real_k = ((b * seq_len + l) * k_heads + hk) * dim + (d_idx * 2);
                    int idx_imag_k = idx_real_k + 1;
                    float k_r = k_in[idx_real_k];
                    float k_i = k_in[idx_imag_k];
                    float mu_k = std::sqrt(k_r * k_r + k_i * k_i);
                    float theta = static_cast<float>(l) * inv_freq[d_idx];
                    float k_phase = theta + delta[d_idx];
                    k_cpp[idx_real_k] = mu_k * std::cos(k_phase);
                    k_cpp[idx_imag_k] = mu_k * std::sin(k_phase);
                }
            }
        }
    }

    // --- 4. 写出 C++ 结果 ---
    save_bin("q_cpp_output.bin", q_cpp.data(), q_cpp.size());
    save_bin("k_cpp_output.bin", k_cpp.data(), k_cpp.size());

    std::cout << "C++ 计算完成并已写出 q_cpp_output.bin/k_cpp_output.bin" << std::endl;

    return 0;
}
