# 本程序为 LIF 神经元的 MPI 分布式仿真，使用了 C 动态库进行加速计算
# Num of Neurons is: 10000
# Num of Niter is 1000
# Time of Process is: 0.08828473091125488

import time
import numpy as np
from mpi4py import MPI
from ctypes import *
import numpy.ctypeslib as ctl

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

ctl_lib = ctl.load_library("test.so", "./")
ctl_lib.lifP.restypes = None
ctl_lib.lifP.argtypes = [
    ctl.ndpointer(dtype=np.float64),
    ctl.ndpointer(dtype='b'), c_int
]

numNeuron = 1000  # 最小集群神经元数量


def process_Neuron(niter):
    # 每个 MPI 进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.float64) * (-70)
    Spike = np.empty(numNeuron, dtype='b')
    SpikeAll = np.empty(comm_size * numNeuron, dtype='b')

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Num of Neurons is:", comm_size * numNeuron)
        print("Num of Niter is", niter)
        start = time.time()

    # 每个进程分别迭代计算，然后全收集
    for i in range(niter):
        ctl_lib.lifP(VmR, Spike, numNeuron)
        comm.Allgather(Spike, SpikeAll)
        # TODO: 绘制单个神经元的膜电位曲线验证仿真正确性

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Time of Process is:", time.time() - start)


# 主程序
if __name__ == '__main__':
    if comm_rank == 0:
        print("本程序为 LIF 神经元的 MPI 分布式仿真，使用了 C 动态库进行加速计算")
    process_Neuron(1000)
