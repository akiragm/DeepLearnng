# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 00:25:06 2023

@author: akira
"""

import numpy as np
A_cpu = np.random.randn(10000, 20000)
B_cpu = np.random.randn(20000, 30000)
# numpy を使ってCPUで行列積を計算する
AB_cpu = np.dot(A_cpu, B_cpu)