# 本程序为 HH 神经元的 MPI 分布式仿真
# 使用 C 动态库进行加速计算
# 神经元数量为： 10000
# 集群个数为： 10
# 迭代次数为： 10000
# Time of Process is: 1.5992639064788818

from ctypes import c_int

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 初始化 C 动态库函数
ctl_lib = ctl.load_library("HH.so", "./HH")
ctl_lib.HH.restypes = None
ctl_lib.HH.argtypes = [
    ctl.ndpointer(dtype=np.float64), c_int,
    ctl.ndpointer(dtype=np.float64),
    ctl.ndpointer(dtype=np.float64),
    ctl.ndpointer(dtype=np.float64),
]


def process_Neuron(niter, numNeuron):
    # 每个 MPI 进程初始化神经元
    VmR = np.zeros(numNeuron, dtype=np.float64)
    n = np.ones(numNeuron, dtype=np.float64) * 0.31
    m = np.ones(numNeuron, dtype=np.float64) * 0.05
    h = np.ones(numNeuron, dtype=np.float64) * 0.59

   # 主进程发送辅助信息
    if comm_rank == 0:
        import os
        import time

        import matplotlib
        import matplotlib.pyplot as plt

        os.system('export DISPLAY=:0')
        matplotlib.use('Agg')
        numPLot = 2000

        print("本程序为 HH 神经元的 MPI 分布式仿真")
        print("使用 C 动态库进行加速计算")
        print("神经元数量为：", comm_size * numNeuron)
        print("集群个数为：", comm_size)
        print("迭代次数为：", niter)
        picU = np.ones(numPLot, dtype=np.float64)  # 绘图数组
        start = time.time()  # 记录仿真时长

    for i in range(niter):
        ctl_lib.HH(VmR, numNeuron, n, m, h)  # 大规模神经元电位计算
        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if (i < numPLot):
                picU[i] = VmR[0]

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Time of Process is:", time.time() - start)
        # 绘制单个神经元的膜电位曲线验证仿真正确性
        x = np.linspace(0, numPLot, numPLot)
        figU, ax = plt.subplots()
        ax.plot(x, picU)
        figU.savefig("./output/HH膜电位.png")


 # 主程序
if __name__ == '__main__':
    # 初始化仿真参数
    numNeurons = 1000  # 最小集群神经元数量
    niters = 10000  # 迭代次数

    process_Neuron(niters, numNeurons)
