#!/bin/bash
# 1. 编译并运行 (PoPE 内部会自动调用 .venv 里的 python)
g++ -O3 -std=c++17 example_PoPE.cpp -o example_pope && ./example_pope

# 2. 如果想单独手动运行对比脚本，可以取消下面这一行的注释：
# /workspaces/PoPE/.venv/bin/python compare_PoPE.py --q_cpp data/q_cpp_output.bin --k_cpp data/k_cpp_output.bin --q_py data/q_py_output.bin --k_py data/k_py_output.bin --batch 1 --seq_len 256 --q_heads 128 --k_heads 1 --dim 64 --tol 1e-5
