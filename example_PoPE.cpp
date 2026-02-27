#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>

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

// --- half (f16) <-> float (f32) conversion helpers ---
static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float f; std::memcpy(&f, &bits, 4); return f;
        }
        // subnormal
        uint32_t m = mant << 13;
        uint32_t e = 0;
        while (!(m & 0x00800000u)) { m <<= 1; e--; }
        uint32_t bits = sign | ((e + 127 - 14) << 23) | (m & 0x007FFFFFu);
        float f; std::memcpy(&f, &bits, 4); return f;
    } else if (exp == 0x1F) {
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f; std::memcpy(&f, &bits, 4); return f;
    }
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

static uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (1 - exp);
        return (uint16_t)(sign | (mant + 0x00001000u) >> 13);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }
    mant = mant + 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0;
        exp++;
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// Load binary file containing f16 (IEEE-754 binary16) values and convert to float
void load_bin_f16(const std::string& path, float* data, size_t count) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) { std::cerr << "无法打开: " << path << std::endl;
    std::filesystem::path current_path = std::filesystem::current_path();
    std::cout << "Warning: Current Path: " << current_path.string() << std::endl;  exit(1); }
    std::streamsize sz = is.tellg();
    if (sz != static_cast<std::streamsize>(count * sizeof(uint16_t))) {
        std::cerr << "警告: 文件大小与预期不符: " << path << std::endl;
    }
    is.seekg(0);
    std::vector<uint16_t> tmp(count);
    is.read(reinterpret_cast<char*>(tmp.data()), count * sizeof(uint16_t));
    if (!is) { std::cerr << "读取错误: " << path << std::endl; exit(1); }
    for (size_t i = 0; i < count; ++i) data[i] = half_to_float(tmp[i]);
}

// Save float buffer as f16 (binary16) on disk
void save_bin_f16(const std::string& path, const float* data, size_t count) {
    std::ofstream os(path, std::ios::binary);
    if (!os) { std::cerr << "无法写入: " << path << std::endl; exit(1); }
    std::vector<uint16_t> tmp(count);
    for (size_t i = 0; i < count; ++i) tmp[i] = float_to_half(data[i]);
    os.write(reinterpret_cast<char*>(tmp.data()), count * sizeof(uint16_t));
}

int main() {
    // --- 1. 调用 Python 生成输入（使用 0-code 目录下的脚本） ---
    // 程序在 0-code 目录下运行，脚本就在当前目录
    std::cout << "Step 1: 正在调用 Python 生成原始矩阵..." << std::endl;
    int ret1 = std::system("/workspaces/PoPE/.venv/bin/python pytorch_PoPE.py");
    if (ret1 != 0) { std::cerr << "调用 Python 生成数据失败, 返回码=" << ret1 << std::endl; return -1; }

    // --- 2. 加载数据 ---
    // 数据存放在 build/data 下
    std::vector<float> q_in(batch * seq_len * q_heads * dim), k_in(batch * seq_len * k_heads * dim);
    std::vector<float> delta(dim / 2, 0.0f), inv_freq(dim / 2);
    load_bin_f16("data/q_input.bin", q_in.data(), q_in.size());
    load_bin_f16("data/k_input.bin", k_in.data(), k_in.size());

    // --- Debug: print first few raw uint16 bits and converted floats from input files ---
    {
        std::ifstream isq("data/q_input.bin", std::ios::binary);
        if (isq) {
            const size_t dbg_n = 8;
            std::vector<uint16_t> raw(dbg_n);
            isq.read(reinterpret_cast<char*>(raw.data()), dbg_n * sizeof(uint16_t));
            std::cout << "data/q_input.bin: first " << dbg_n << " uint16 raw: ";
            for (size_t i = 0; i < dbg_n; ++i) std::cout << raw[i] << " ";
            std::cout << std::endl << "data/q_input.bin: first " << dbg_n << " as float: ";
            for (size_t i = 0; i < dbg_n; ++i) std::cout << half_to_float(raw[i]) << " ";
            std::cout << std::endl;
        }
        std::ifstream isk("data/k_input.bin", std::ios::binary);
        if (isk) {
            const size_t dbg_n = 8;
            std::vector<uint16_t> raw(dbg_n);
            isk.read(reinterpret_cast<char*>(raw.data()), dbg_n * sizeof(uint16_t));
            std::cout << "data/k_input.bin: first " << dbg_n << " uint16 raw: ";
            for (size_t i = 0; i < dbg_n; ++i) std::cout << raw[i] << " ";
            std::cout << std::endl << "data/k_input.bin: first " << dbg_n << " as float: ";
            for (size_t i = 0; i < dbg_n; ++i) std::cout << half_to_float(raw[i]) << " ";
            std::cout << std::endl;
        }
    }

    // Python 端通常没有单独保存 inv_freq/delta，这里按 PoPE 的公式重建, delta 默认为 0
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
    save_bin_f16("data/q_cpp_output.bin", q_cpp.data(), q_cpp.size());
    save_bin_f16("data/k_cpp_output.bin", k_cpp.data(), k_cpp.size());

    std::cout << "C++ 计算完成并已写出 q_cpp_output.bin/k_cpp_output.bin" << std::endl;

    // 调用对比脚本
    std::string cmp_cmd = "/workspaces/PoPE/.venv/bin/python compare_PoPE.py --q_cpp data/q_cpp_output.bin --k_cpp data/k_cpp_output.bin --q_py data/q_py_output.bin --k_py data/k_py_output.bin --batch 1 --seq_len 256 --q_heads 128 --k_heads 1 --dim 64 --tol 1e-5";
    std::cout << "Running compare: " << cmp_cmd << std::endl;
    int cmp_rc = std::system(cmp_cmd.c_str());
    if (cmp_rc != 0) {
        std::cerr << "Compare script returned non-zero: " << cmp_rc << std::endl;
        return cmp_rc;
    }

    return 0;
}
