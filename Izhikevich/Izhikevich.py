import os
from ctypes import c_float, c_int

import numpy as np
import numpy.ctypeslib as ctl
from mpi4py import MPI

# 初始化 MPI 配置
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
singlecomm_size = comm.Get_size()

# 初始化 C 动态库函数
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
ctl_lib = ctl.load_library("Izhikevich.so", father_path)
ctl_lib.rungeKutta.restypes = None
ctl_lib.rungeKutta.argtypes = [
    ctl.ndpointer(dtype=np.single), 
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype='b'), c_float,
    c_float, c_float, c_float, c_int
]

def process_Neuron(niter: int, numNeuron: int, totalNeuron: int):
    # 抑制型
    # a = 0.02; b = 0.2; c = -65.0; d = 6  
    # 兴奋型
    a = 0.2; b = 0.26; c = -65.0; d = 0
    Vm = np.ones(numNeuron, dtype=np.single) * (-55)
    u = np.ones(numNeuron, dtype=np.single) * (-0)
    Ij = ((np.random.rand(numNeuron, totalNeuron)) * (5)).astype(np.single)

    # 主进程
    if comm_rank == 0:
        import time

        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        numPlot = 1000
        # 打印发送辅助信息
        print("本程序为 Izhikevich 神经元的 MPI 分布式仿真")
        print("Linux MPI 版本为")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", numNeuron)
        print("迭代次数为：", niter)
        # 绘图数组
        picU = np.ones(numPlot, dtype=np.single)
        picS = np.zeros((numPlot, totalNeuron), dtype=bool)
        picF = np.zeros(numPlot, dtype=np.int32)
        start = time.time()  # 记录仿真时长

    for i in range(niter):
        Ij = ((np.random.rand(numNeuron, totalNeuron)) * (5)).astype(np.single)
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(totalNeuron, dtype='b')
        ctl_lib.rungeKutta(Vm, u, Ij, Spike, a, b, c, d, numNeuron)
        comm.Allgather(Spike, SpikeAll)
        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if (i < numPlot):
                picU[i] = Vm[5]
                picS[i] = SpikeAll
                picF[i] = sum(SpikeAll)
    
        # 主进程发送辅助信息
    if comm_rank == 0:
        print("运行时间为:", time.time() - start)

        # 绘制单个神经元的膜电位曲线验证仿真正确性
        x = np.linspace(0, numPlot, numPlot)
        ax = plt.subplot(1,3,1)
        ax.plot(x, picU)
        # fig.savefig(os.path.join(father_path, "Izhikevich 膜电位.png"))
        ay = plt.subplot(1,3,2)
        for i in range(numPlot):
            y = np.argwhere(picS[i] == 1)
            x = np.ones(len(y)) * i
            ay.scatter(x, y, c='black', s=0.5)
        # fig.savefig(os.path.join(father_path, "Izhikevich 放电栅格.png"))
        af = plt.subplot(1,3,3)
        x = np.linspace(0, numPlot, numPlot)
        af.plot(x, picF)
        fig.savefig(os.path.join(father_path, "Izhikevich 放电率.png"))

if __name__ == '__main__':
    # 初始化仿真参数
    if comm_rank == 0:
        numNeurons = 500
    else:
        numNeurons = 500
        
    totalNeurons = comm.allreduce(numNeurons)
    niters = 1000  # 迭代次数
    process_Neuron(niters, numNeurons, totalNeurons)
