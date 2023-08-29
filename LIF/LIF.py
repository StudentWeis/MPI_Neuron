import os
import time
from ctypes import c_int

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ctypeslib as ctl
import psutil

# 初始化 C 动态库函数
ctl_lib = ctl.load_library("LIF.so", "./LIF")
ctl_lib.lifPI.restypes = None
ctl_lib.lifPI.argtypes = [
    ctl.ndpointer(dtype=np.single), ctl.ndpointer(dtype='b'), c_int,
    ctl.ndpointer(dtype=np.single), ctl.ndpointer(dtype=np.int8)
]
ctl_lib.IjDot.restypes = None
ctl_lib.IjDot.argtypes = [
    ctl.ndpointer(dtype=np.single), ctl.ndpointer(dtype='b'),
    c_int, c_int, ctl.ndpointer(dtype=np.single)
]


def process_Neuron(niter, numNeuron):

    # 每个 MPI 进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.single) * (-70)
    Ij = np.ones(numNeuron, dtype=np.single) * (0.25)
    # 初始化突触
    # 兴奋型连接和抑制型连接
    WeightMask = (np.random.choice([-1, 1, 0], size=(numNeuron, numNeuron), p=[.2, .2, .6])).astype(np.int8)
    WeightRand = (np.random.rand(numNeuron, numNeuron) * 0.1).astype(np.single)
    WeightRand = np.multiply(WeightMask, WeightRand).astype(np.single)
    # 释放内存占用
    del WeightMask
    # 初始化灭火期记录
    period = np.zeros(numNeuron, dtype=np.int8)

    matplotlib.use('Agg')
    numPlot = 100
    # 打印发送辅助信息
    print("本程序为 LIF 神经元的 NoMPI 仿真")
    print("考虑神经元的灭火期")
    print("通过突触全随机连接计算电流")
    print("使用 C 动态库进行加速计算")
    print("使用 Matplotlib 绘图")
    print("神经元数量为：", numNeuron)
    print("迭代次数为：", niter)
    # 绘图数组
    picU = np.ones(numPlot, dtype=np.single)
    picS = np.zeros((numPlot, numNeurons), dtype=bool)
    picF = np.zeros(numPlot, dtype=np.int32)
    start = time.time()  # 记录仿真时长

    for i in range(niter):
        # 初始化放电矩阵
        Spike = np.zeros(numNeuron, dtype='b')
        ctl_lib.lifPI(VmR, Spike, numNeuron, Ij, period)  # 大规模神经元电位计算
        ctl_lib.IjDot(WeightRand, Spike, numNeuron, 1, Ij)  # 计算突触电流

        # 记录单个神经元的膜电位数据
        if (i < numPlot):
            picU[i] = VmR[90]
            picS[i] = Spike
            picF[i] = sum(Spike)

    print("运行时间为:", time.time() - start)
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

    # 绘制单个神经元的膜电位曲线验证仿真正确性
    x = np.linspace(0, numPlot, numPlot)
    figU, ax = plt.subplots()
    ax.plot(x, picU)
    figU.savefig("./output/NoMPI-LIF膜电位.png")
    figS, ay = plt.subplots()
    for i in range(numPlot):
        y = np.argwhere(picS[i] == 1)
        x = np.ones(len(y)) * i
        ay.scatter(x, y, c='black', s=0.5)
    figS.savefig("./output/NoMPI-LIF放电栅格图.png")
    figF, af = plt.subplots()
    x = np.linspace(0, numPlot, numPlot)
    af.plot(x, picF)
    figF.savefig("./output/NoMPI-LIF放电率.png")


# 主程序
if __name__ == '__main__':
    # 初始化仿真参数
    numNeurons = 10000  # 最小集群神经元数量
    niters = 1000  # 迭代次数
    process_Neuron(niters, numNeurons)
