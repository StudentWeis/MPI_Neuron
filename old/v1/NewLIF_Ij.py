# 本程序为 LIF 神经元的 MPI 分布式仿真
# 通过突触全随机连接计算电流
# 使用 C 动态库进行加速计算
# 使用 Matplotlib 绘图
# 神经元数量为： 10000
# 集群进程数：10
# 迭代次数为 1000
# Time of Process is: 0.0956885814666748

import time
from ctypes import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

matplotlib.use('Agg')

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 初始化 C 动态库函数
ctl_lib = ctl.load_library("test.so", "./")
ctl_lib.lifPI.restypes = None
ctl_lib.lifPI.argtypes = [
    ctl.ndpointer(dtype=np.float64),
    ctl.ndpointer(dtype='b'), c_int,
    ctl.ndpointer(dtype=np.float64)
]
ctl_lib.IjDot.restypes = None
ctl_lib.IjDot.argtypes = [
    ctl.ndpointer(dtype=np.float64),
    ctl.ndpointer(dtype='b'), c_int, c_int,
    ctl.ndpointer(dtype=np.float64)
]

# 初始化仿真参数
numNeurons = 1000  # 最小集群神经元数量
niters = 1000  # 迭代次数


def process_Neuron(niter, numNeuron):

    # 每个 MPI 进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.float64) * (-70)
    Ij = np.ones(numNeuron, dtype=np.float64) * (0.25)
    # 初始化突触
    WeightMask = np.random.choice(
        [-1, 1, 0], size=(numNeuron, numNeuron * comm_size), p=[.2, .2, .6]) * (0.1)
    WeightRand = np.random.rand(numNeuron, numNeuron * comm_size)
    Weight = WeightMask * WeightRand

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("本程序为 LIF 神经元的 MPI 分布式仿真")
        print("通过突触全随机连接计算电流")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", comm_size * numNeuron)
        print("迭代次数为", niter)
        pic = np.ones(50, dtype=np.float64)  # 绘图数组
        start = time.time()

    for i in range(niter):
        # 每个进程分别迭代计算，然后全收集
        # 初始化放电矩阵
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(comm_size * numNeuron, dtype='b')
        ctl_lib.lifPI(VmR, Spike, numNeuron, Ij)  # 使用 C 动态库进行大规模神经元电位计算
        comm.Allgather(Spike, SpikeAll)  # 全收集
        ctl_lib.IjDot(Weight, SpikeAll, numNeuron,
                      comm_size, Ij)  # 使用 C 动态库计算突触电流

        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if i < 50:
                pic[i] = VmR[0]
                # print(SpikeAll)

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Time of Process is:", time.time() - start)

        # 绘制单个神经元的膜电位曲线验证仿真正确性
        x = np.linspace(0, 50, 50)
        fig, ax = plt.subplots()
        ax.plot(x, pic, linewidth=2.0)
        fig.savefig("./ts.png")


# 主程序
if __name__ == '__main__':
    process_Neuron(niters, numNeurons)
