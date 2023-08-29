from ctypes import c_float, c_int
import os

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
singlecomm_size = comm.Get_size()

# 初始化 C 动态库函数
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
ctl_lib = ctl.load_library("Izhikevich.so", father_path)
ctl_lib.rungeKutta.restypes = None
ctl_lib.rungeKutta.argtypes = [
    ctl.ndpointer(dtype=np.single), 
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype=np.single),
    c_float, c_float, c_float, c_float, c_int
]

def main(niter, numNeuron):
    a = 0.05  # 恢复变量的时间尺度，越小，恢复越慢
    b = 0.2  # 恢复变量依赖膜电位的阈值下随机波动的敏感度
    c = -50.0  # 膜电位复位值
    d = 5  # 恢复变量复位值
    Vm = np.ones(numNeuron, dtype=np.single) * (-50)
    u = np.ones(numNeuron, dtype=np.single) * (0)
    Ij = np.ones(numNeuron, dtype=np.single) * (5)
    for i in range(niter):  # 模拟时长(ms)
        ctl_lib.rungeKutta(Vm, u, Ij, a, b, c, d, numNeuron)


main(1000, 10)
