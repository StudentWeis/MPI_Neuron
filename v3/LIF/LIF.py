# 本程序为 LIF 神经元的 MPI 分布式仿真
# 考虑神经元的灭火期
# 通过突触全随机连接计算电流
# 使用 C 动态库进行加速计算
# 使用 Matplotlib 绘图
# 神经元数量为： 10000
# 集群个数为： 10
# 迭代次数为： 1000
# Time of Process is: 0.11419963836669922

from ctypes import c_int

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 初始化 C 动态库函数
ctl_lib = ctl.load_library("LIF.so", "./LIF")
ctl_lib.lifPI.restypes = None
ctl_lib.lifPI.argtypes = [
    ctl.ndpointer(dtype=np.float64), ctl.ndpointer(dtype='b'), c_int,
    ctl.ndpointer(dtype=np.float64), ctl.ndpointer(dtype=np.int8)
]
ctl_lib.IjDot.restypes = None
ctl_lib.IjDot.argtypes = [
    ctl.ndpointer(dtype=np.float64), ctl.ndpointer(dtype='b'),
    c_int, c_int, ctl.ndpointer(dtype=np.float64)
]


def process_Neuron(niter, numNeuron):

    # 每个 MPI 进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.float64) * (-70)
    Ij = np.ones(numNeuron, dtype=np.float64) * (0.25)
    # 初始化突触
    WeightMask = np.random.choice(  # 兴奋型连接和抑制型连接
        [-1, 1, 0], size=(numNeuron, numNeuron * comm_size), p=[.2, .2, .6]) * (0.1)
    WeightRand = np.random.rand(numNeuron, numNeuron * comm_size)
    Weight = WeightMask * WeightRand
    # 初始化灭火期记录
    period = np.zeros(numNeuron, dtype=np.int8)

    # 主进程
    if comm_rank == 0:
        import time
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        numPlot = 100
        # 打印发送辅助信息
        print("本程序为 LIF 神经元的 MPI 分布式仿真")
        print("Linux MPI 版本为")
        print("考虑神经元的灭火期")
        print("通过突触全随机连接计算电流")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", comm_size * numNeuron)
        print("集群个数为：", comm_size)
        print("迭代次数为：", niter)
        # 绘图数组
        picU = np.ones(numPlot, dtype=np.float64)
        picS = np.zeros((numPlot, comm_size * numNeurons), dtype=bool)
        picF = np.zeros(numPlot, dtype=np.int32)
        start = time.time()  # 记录仿真时长

    for i in range(niter):
        # 每个进程分别迭代计算，然后全收集
        # 初始化放电矩阵
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(comm_size * numNeuron, dtype='b')
        ctl_lib.lifPI(VmR, Spike, numNeuron, Ij, period)  # 大规模神经元电位计算
        comm.Allgather(Spike, SpikeAll)  # 全收集放电矩阵
        ctl_lib.IjDot(Weight, SpikeAll, numNeuron, comm_size, Ij)  # 计算突触电流

        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if (i < numPlot):
                picU[i] = VmR[90]
                picS[i] = SpikeAll
                picF[i] = sum(SpikeAll)

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("运行时间为:", time.time() - start)

        # 绘制单个神经元的膜电位曲线验证仿真正确性
        x = np.linspace(0, numPlot, numPlot)
        figU, ax = plt.subplots()
        ax.plot(x, picU)
        figU.savefig("./output/LIF膜电位.png")
        figS, ay = plt.subplots()
        for i in range(numPlot):
            y = np.argwhere(picS[i] == 1)
            x = np.ones(len(y)) * i
            ay.scatter(x, y, c='black', s=0.5)
        figS.savefig("./output/LIF放电栅格图.png")
        figF, af = plt.subplots()
        x = np.linspace(0, numPlot, numPlot)
        af.plot(x, picF)
        figF.savefig("./output/LIF放电率.png")


# 主程序
if __name__ == '__main__':
    # 初始化仿真参数
    numNeurons = 1000  # 最小集群神经元数量
    niters = 1000  # 迭代次数
    process_Neuron(niters, numNeurons)
