# HH 神经元 MPI 仿真
import os
from ctypes import c_int

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 初始化 C 动态库函数
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
ctl_lib = ctl.load_library("HH.so", father_path)
ctl_lib.HH.restypes = None
ctl_lib.HH.argtypes = [
    ctl.ndpointer(dtype=np.single), 
    ctl.ndpointer(dtype='b'),
    ctl.ndpointer(dtype='b'), c_int,
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype=np.single),
]

# 神经元进程
def processNeuron(niter: int, numNeuron: int, totalNeuron: int):
    # 所有进程初始化神经元
    Vm = np.zeros(numNeuron, dtype=np.single)
    n = ((np.random.rand(numNeuron)) * (0.31)).astype(np.single)
    m = ((np.random.rand(numNeuron)) * (0.05)).astype(np.single)
    h = ((np.random.rand(numNeuron)) * (0.59)).astype(np.single)
    period = np.zeros(numNeuron, dtype='b') # 放电标志位

    # 主进程提供辅助信息
    if comm_rank == 0:
        # Matplotlib 画图记录变量
        numPlot = 1000
        picV = np.ones(numPlot, dtype=np.single)
        picY = np.zeros((numPlot, totalNeuron), dtype=bool)
        picF = np.zeros(numPlot, dtype=np.single)

        # 打印辅助信息
        print("本程序为 HH 神经元的 MPI 分布式仿真")
        print("Linux MPI 版本为")
        print("随机初始化电导率")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", totalNeuron)
        print("集群个数为：", comm_size)
        print("迭代次数为：", niter)

        # 记录仿真时长
        import time
        start = time.time()  

    # 所有进程完成迭代
    for i in range(niter):
        # 每个进程分别迭代计算，然后全收集
        # 初始化放电矩阵
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(totalNeuron, dtype='b')
        ctl_lib.HH(Vm, Spike, period, numNeuron, n, m, h)  # 大规模神经元电位计算
        comm.Allgather(Spike, SpikeAll)

        # 记录单个神经元的实验数据
        if comm_rank == 0:
            if (i < numPlot):
                picV[i] = Vm[5]
                picY[i] = SpikeAll
                picF[i] = sum(SpikeAll)/totalNeuron*100

    # 主进程提供辅助信息
    if comm_rank == 0:
        print("运行时间为：", time.time() - start)

        # 绘制单个神经元实验结果验证仿真正确性
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.style.use('bmh')

        # HH Simulation results
        fig = plt.figure(dpi=300)
        x = np.linspace(0, numPlot, numPlot)
        # 膜电位
        av = plt.subplot(3,1,1)
        av.plot(x, picV)
        av.axes.xaxis.set_ticklabels([])
        av.set_title("(c1)", x=1.05, y=0.8, size=10) # Membrane Potential
        av.set_ylabel("Voltage/(mV)")
        av.set_xlim(100,numPlot)
        # 放电率
        af = plt.subplot(3,1,2)
        af.plot(x, picF)
        af.axes.xaxis.set_ticklabels([])
        af.set_title("(c2)", x=1.05, y=0.8, size=10) # Firing Rate
        af.set_ylabel("Firing Rate/(%)", labelpad=13.5)
        af.set_xlim(100,numPlot)
        # 放电栅格
        ay = plt.subplot(3,1,3)
        for i in range(numPlot):
            y = np.argwhere(picY[i] == 1)
            x = np.ones(len(y)) * i
            ay.scatter(x, y, c='black', s=0.5)
        ay.set_title("(c3)", x=1.05, y=0.8, size=10) # Firing Grid Map
        ay.set_ylabel("Neuron No.")
        ay.set_xlim(100,numPlot)
        # 保存图片
        fig.savefig(os.path.join(father_path, "HH.png"))


 # 主程序
if __name__ == '__main__':
    # 初始化仿真参数
    if comm_rank == 0:
        numNeurons = 100
    else:
        numNeurons = 100
        
    totalNeurons = comm.allreduce(numNeurons)
    niters = 1000  # 迭代次数
    processNeuron(niters, numNeurons, totalNeurons)
