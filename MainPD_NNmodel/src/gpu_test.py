# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:51:08 2023

@author: akira
"""

import cupy as cp  # cupy は cp と略すのが一般的なようです

# np.ndarray から GPU 上のメモリにデータを移動する
A_gpu = cp.ndarray(A_cpu)
B_gpu = cp.ndarray(B_cpu)

# cupy を使って GPU で行列積を計算する
AB_gpu = cp.dot(A_gpu, B_gpu)

# メインメモリ上にデータを移動する
AB_cpu2 = AB_gpu.get()  # AB_cpu2 は np.ndarray 型