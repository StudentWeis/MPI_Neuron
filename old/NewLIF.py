# 本程序为 LIF 神经元的 MPI 分布式仿真，使用了 C 动态库进行加速计算，加入了放电数组，使用 Matplotlib 绘图
# Num of Neurons is: 10000
# Num of Niter is 1000
# Time of Process is: 0.10401415824890137

import time
from ctypes import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

matplotlib.use('Agg')

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

ctl_lib = ctl.load_library("test.so", "./")
ctl_lib.lifPI.restypes = None
ctl_lib.lifPI.argtypes = [
    ctl.ndpointer(dtype=np.float64),
    ctl.ndpointer(dtype='b'), c_int,
    ctl.ndpointer(dtype=np.float64)
]

numNeuron = 1000  # 最小集群神经元数量


def process_Neuron(niter):

    # 每个 MPI 进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.float64) * (-70)
    Ij = np.ones(numNeuron, dtype=np.float64) * (0.25)

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Num of Neurons is:", comm_size * numNeuron)
        print("Num of Niter is", niter)
        pic = np.ones(50, dtype=np.float64)  # 绘图数组
        start = time.time()

    # 每个进程分别迭代计算，然后全收集
    for i in range(niter):
        # 复原放电情况
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(comm_size * numNeuron, dtype='b')
        ctl_lib.lifPI(VmR, Spike, numNeuron, Ij)  # 使用 C 动态库进行大规模矩阵计算
        comm.Allgather(Spike, SpikeAll)  # 全收集
        # 记录单个神经元的膜电位数据
        if i < 50:
            if comm_rank == 0:
                pic[i] = VmR[0]

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
    if comm_rank == 0:
        print("本程序为 LIF 神经元的 MPI 分布式仿真，使用了 C 动态库进行加速计算，加入了放电数组，使用 Matplotlib 绘图")
    process_Neuron(1000)
