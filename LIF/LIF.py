# LIF 神经元 MPI 仿真
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
ctl_lib = ctl.load_library("LIF.so", father_path)
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

# 神经元进程
def processNeuron(niter: int, numNeuron: int, totalNeuron: int):

    # 同步，并计算各个进程神经元数量以及分布情况
    count = np.zeros(comm_size, dtype=int)
    temp = np.ones(1, dtype=int) * numNeuron
    comm.Allgather(temp, count)
    display = np.roll(count, 1)
    display[0] = 0
    display = display.cumsum()

    # 所有进程初始化神经元
    VmR = np.ones(numNeuron, dtype=np.single) * (-70)
    Ij = np.ones(numNeuron, dtype=np.single) * (0.25)
    period = np.zeros(numNeuron, dtype=np.int8) # 灭火期记录

    # 交错生成 Weight，缓解内存问题
    MaskOKRecv = False
    MaskOKSend = False
    
    # 主进程提供辅助信息
    if comm_rank == 0:
        import psutil
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    # 其他进程依次等待上一进程完毕
    else:
        MaskOKRecv = comm.recv(source=comm_rank-1)
        
    # 初始化突触
    WeightMask = np.random.choice([-1, 1, 0], size=(numNeuron, totalNeuron), p=[.2, .2, .6]).astype(np.int8)
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
    
    # 初始化完成后依次向后一进程发送完毕指令
    if comm_size > 1:
        if comm_rank == comm_size-1:
            comm.ssend(MaskOKSend, dest=0)
        else:
            comm.ssend(MaskOKSend, dest=comm_rank+1)
        # 主进程等待最后一个进程完毕
        if comm_rank == 0:
            MaskOKRecv = comm.recv(source=comm_size-1)
            if(MaskOKRecv):
                print("交错式初始化内存任务完毕")

    # 主进程提供辅助信息
    if comm_rank == 0:
        # Matplotlib 画图记录变量
        numPlot = 100
        startPlot = 100
        picV = np.ones(numPlot, dtype=np.single)
        picY = np.zeros((numPlot, totalNeuron), dtype=bool)
        picF = np.zeros(numPlot, dtype=np.single)
        
        # 打印辅助信息
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
        print("仿真步长为 1ms")
        print("迭代次数为：", niter)
        print("理论时间为：", niter/1000)

        # 记录仿真时长
        import time
        start = time.time()
        gathertime = 0

    # 所有进程完成迭代
    for i in range(niter):
        # 每个进程分别迭代计算，然后全收集
        # 初始化放电矩阵
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(totalNeuron, dtype='b')
        ctl_lib.lifPI(VmR, Spike, numNeuron, Ij, period)  # 大规模神经元电位计算
        if comm_rank == 0:
            start2 = time.time()  # 收集仿真时长
        # comm.Allgatherv(Spike, [SpikeAll, count, display, MPI.BYTE])
        comm.Allgather(Spike, SpikeAll)
        if comm_rank == 0:
            gathertime += time.time() - start2
        ctl_lib.IjDot(WeightRand, SpikeAll, numNeuron, totalNeuron, Ij)  # 计算突触电流
        # Ij = ((np.random.rand(numNeuron, totalNeuron)) * (0.5)).astype(np.single)

        # 记录单个神经元的实验数据
        if comm_rank == 0:
            if ((startPlot <+ i) & (i < startPlot + numPlot)):
                picV[i - startPlot] = VmR[90]
                picY[i - startPlot] = SpikeAll
                picF[i - startPlot] = sum(SpikeAll)/totalNeuron*100

    # 主进程发送辅助信息
    if comm_rank == 0:

        print("运行时间为:", time.time() - start)
        print("收集时间为:", gathertime)

        np.save('lifV', picV)
        np.save('lifS', picY)

        # 绘制单个神经元实验结果验证仿真正确性
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.style.use('bmh')

        # LIF Simulation results
        fig = plt.figure(dpi=500, figsize=(1.5,4.5))
        x = np.linspace(0, numPlot, numPlot)
        # 膜电位
        av = plt.subplot(3,1,1)
        av.plot(x, picV)
        av.axes.xaxis.set_ticklabels([])
        av.set_title("(a1)", x=1.05, y=0.8, size=10) # Membrane Potential
        av.set_ylabel("Voltage/mV")
        # av.set_xlim(50,150)
        # 放电率
        af = plt.subplot(3,1,2)
        af.plot(x, picF)
        af.axes.xaxis.set_ticklabels([])
        af.set_title("(a2)", x=1.05, y=0.8, size=10) # Firing Rate
        af.set_ylabel("Firing Rate/(%)", labelpad=11)
        # af.set_xlim(50,150)
        # 放电栅格
        ay = plt.subplot(3,1,3)
        for i in range(numPlot):
            y = np.argwhere(picY[i] == 1)
            x = np.ones(len(y)) * i
            ay.scatter(x, y, c='black', s=0.004)
        ay.set_title("(a3)", x=1.05, y=0.8, size=10) # Firing Grid Map
        ay.set_ylabel("Neuron No.")
        # ay.set_xlim(50,150)
        # 保存图片
        fig.savefig(os.path.join(father_path, "LIF.png"))


if __name__ == '__main__':
    # 初始化仿真参数
    if comm_rank == 0:
        numNeurons = 2500
    else:
        numNeurons = 2500
        
    totalNeurons = comm.allreduce(numNeurons)
    niters = 1000  # 迭代次数
    processNeuron(niters, numNeurons, totalNeurons)
