from ctypes import c_double

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 初始化 C 动态库函数
ctl_lib = ctl.load_library("Izhikevich.so", "./Izhikevich")
ctl_lib.IRungeKutta.restypes = None
ctl_lib.IRungeKutta.argtypes = [
    ctl.ndpointer(dtype=np.float64), 
    ctl.ndpointer(dtype=np.float64),
    c_double, c_double, c_double,
    c_double, c_double
]


a = 0.05  # 恢复变量的时间尺度，越小，恢复越慢
b = 0.2  # 恢复变量依赖膜电位的阈值下随机波动的敏感度
c = -50.0  # 膜电位复位值
d = 5  # 恢复变量复位值


def main():
    v = -65  # 初始膜电位
    u = -0  # 初始恢复量
    t = 0.1  # 时间步长
    Ij = 5  # 输入电流
    vs = []  # 记录v
    us = []  # 记录u
    time = []  # 时间戳
    for i in np.arange(0, 1000, t):  # 模拟时长(ms)
        time.append(i)
        ctl_lib.IRungeKutta(v, u, Ij, a, b, c, d)
        if v >= 30:  # 发射脉冲
            vs.append(30)
            v = c
            u = u+d
        else:
            vs.append(v)
        us.append(u)
    plt.plot(time, vs)
    plt.savefig("test.png")


main()
