from ctypes import c_int

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 初始化 C 动态库函数
ctl_lib = ctl.load_library("LIF_Nums.so", "./LIF")
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

def process_Neuron(niter: int, numNeuron: int, totalNeuron: int):

    # 同步，并计算各个进程神经元数量以及分布情况
    count = np.zeros(comm_size, dtype=int)
    temp = np.ones(1, dtype=int) * numNeuron
    comm.Allgather(temp, count)
    display = np.roll(count, 1)
    display[0] = 0
    display = display.cumsum()

    # 每个 MPI 进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.single) * (-70)
    Ij = np.ones(numNeuron, dtype=np.single) * (0.25)

    # 交错生成 Weight，缓解内存问题
    MaskOKRecv = False
    MaskOKSend = False
    
    if comm_rank == 0:
        import psutil
        import os
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    else:
        MaskOKRecv = comm.recv(source=comm_rank-1)
        
    # 初始化突触，兴奋型连接和抑制型连接
    WeightMask = np.random.choice([-1, 1, 0], size=(numNeuron, totalNeuron), p=[.2, .2, .6])
    if comm_rank == 0:
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    WeightRand = ((np.random.rand(numNeuron, totalNeuron)) * (0.1)).astype(np.single)
    if comm_rank == 0:
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    WeightRand = np.multiply(WeightMask, WeightRand).astype(np.single)
    if comm_rank == 0:
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    # 释放内存占用
    del WeightMask
    if comm_rank == 0:
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    if comm_rank == comm_size-1:
        comm.ssend(MaskOKSend, dest=0)
    else:
        comm.ssend(MaskOKSend, dest=comm_rank+1)
    if comm_rank == 0:
        MaskOKRecv = comm.recv(source=comm_size-1)
        if(MaskOKRecv):
            print("交错式初始化内存任务完毕")

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
        print("交错式内存初始化+同步启动")
        print("灵活调节神经元个数")
        print("考虑神经元的灭火期")
        print("通过突触全随机连接计算电流")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", totalNeuron)
        print("集群个数为：", comm_size)
        print("迭代次数为：", niter)
        # 绘图数组
        picU = np.ones(numPlot, dtype=np.single)
        picS = np.zeros((numPlot, totalNeuron), dtype=bool)
        picF = np.zeros(numPlot, dtype=np.int32)
        start = time.time()  # 记录仿真时长
        gathertime = 0

    for i in range(niter):
        # 每个进程分别迭代计算，然后全收集
        # 初始化放电矩阵
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(totalNeuron, dtype='b')
        ctl_lib.lifPI(VmR, Spike, numNeuron, Ij, period)  # 大规模神经元电位计算
        if comm_rank == 0:
            start2 = time.time()  # 记录仿真时长
        comm.Allgatherv(Spike, [SpikeAll, count, display, MPI.BYTE])
        # comm.Allgather(Spike, SpikeAll)
        if comm_rank == 0:
            gathertime += time.time() - start2
        ctl_lib.IjDot(WeightRand, SpikeAll, numNeuron, totalNeuron, Ij)  # 计算突触电流

        # 记录单个神经元的膜电位数据
        # if comm_rank == 0:
        #     if (i < numPlot):
        #         picU[i] = VmR[90]
        #         picS[i] = SpikeAll
        #         picF[i] = sum(SpikeAll)

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("运行时间为:", time.time() - start)
        print("收集时间为:", gathertime)

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


if __name__ == '__main__':
    # 初始化仿真参数
    if comm_rank == 0:
        numNeurons = 7000
    else:
        numNeurons = 7000
    totalNeurons = comm.allreduce(numNeurons)
    niters = 1000  # 迭代次数
    process_Neuron(niters, numNeurons, totalNeurons)
