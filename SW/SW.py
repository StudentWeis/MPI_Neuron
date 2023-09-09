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
ctl_lib = ctl.load_library("SW.so", father_path)
ctl_lib.SW.restypes = None
ctl_lib.SW.argtypes = [
    ctl.ndpointer(dtype=np.single), 
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype='b'), c_int
]
ctl_lib.IjDot.restypes = None
ctl_lib.IjDot.argtypes = [
    ctl.ndpointer(dtype=np.single), 
    ctl.ndpointer(dtype='b'),
    ctl.ndpointer(dtype='b'), c_int, c_int,
    ctl.ndpointer(dtype=np.single)
]

def process_Neuron(niter: int, numNeuron: int, totalNeuron: int):

    # 初始化突触
    WeightMask = np.random.choice([-1, 1, 0], size=(numNeuron, totalNeuron), p=[.2, .2, .6]).astype(np.int8)
    WeightRand = ((np.random.rand(numNeuron, totalNeuron)) * (1)).astype(np.single)
    WeightRand = np.multiply(WeightMask, WeightRand).astype(np.single)
    del WeightMask # 释放内存占用

    # 初始化神经元
    Vm = np.ones(numNeuron, dtype=np.single) * (-55)
    u = np.ones(numNeuron, dtype=np.single) * (-0)
    Ij = ((np.random.rand(numNeuron, totalNeuron)) * (5)).astype(np.single)

    # 神经元分类
    ClassNeuron = np.random.choice([0, 1], size=(numNeuron), p=[.2, .8]).astype(bool)
    

    # 主进程
    if comm_rank == 0:
        import time

        # 绘图数组
        numPlot = 500
        picV = np.ones(numPlot, dtype=np.single)
        picY = np.zeros((numPlot, totalNeuron), dtype=bool)
        picF = np.zeros(numPlot, dtype=np.single)
        
        # 打印发送辅助信息
        print("本程序为 Izhikevich 神经元的 MPI 分布式仿真")
        print("Linux MPI 版本为")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", totalNeuron)
        print("迭代次数为：", niter)

        # 记录仿真时长
        start = time.time()  

    for i in range(niter):
        Spike = np.zeros(numNeuron, dtype='b')
        SpikeAll = np.zeros(totalNeuron, dtype='b')
        ctl_lib.SW(Vm, u, Ij, Spike, ClassNeuron, numNeuron)
        comm.Allgather(Spike, SpikeAll)
        ctl_lib.IjDot(WeightRand, SpikeAll, numNeuron, totalNeuron, Ij)  # 计算突触电流

        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if (i < numPlot):
                picV[i] = Vm[5]
                picY[i] = SpikeAll
                picF[i] = sum(SpikeAll)/totalNeuron*100
    
    # 主进程发送辅助信息
    if comm_rank == 0:
        print("运行时间为:", time.time() - start)

        # 绘制单个神经元实验结果验证仿真正确性
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.style.use('bmh')

        # Izhikevich Simulation results
        fig = plt.figure(dpi=300)
        x = np.linspace(0, numPlot, numPlot)
        # 膜电位
        av = plt.subplot(3,1,1)
        av.plot(x, picV)
        av.axes.xaxis.set_ticklabels([])
        av.set_title("(b1)", x=1.05, y=0.8, size=10) # Membrane Potential
        av.set_ylabel("Voltage/(mV)")
        av.set_xlim(100,500)
        # 放电率
        af = plt.subplot(3,1,2)
        af.plot(x, picF)
        af.axes.xaxis.set_ticklabels([])
        af.set_title("(b2)", x=1.05, y=0.8, size=10) # Firing Rate
        af.set_ylabel("Firing Rate/(%)", labelpad=13.5)
        af.set_xlim(100,500)
        # 放电栅格
        ay = plt.subplot(3,1,3)
        for i in range(numPlot):
            y = np.argwhere(picY[i] == 1)
            x = np.ones(len(y)) * i
            ay.scatter(x, y, c='black', s=0.5)
        ay.set_title("(b3)", x=1.05, y=0.8, size=10) # Firing Grid Map
        ay.set_ylabel("Neuron No.")
        ay.set_xlim(100,500)
        # 保存图片
        fig.savefig(os.path.join(father_path, "Izhikevich.png"))

if __name__ == '__main__':
    # 初始化仿真参数
    if comm_rank == 0:
        numNeurons = 625
    else:
        numNeurons = 625
    totalNeurons = comm.allreduce(numNeurons)
    niters = 1000  # 迭代次数
    process_Neuron(niters, numNeurons, totalNeurons)
