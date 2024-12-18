import os
from ctypes import c_int

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
    ctl.ndpointer(dtype='b'), 
    ctl.ndpointer(dtype=c_int), c_int
]
ctl_lib.IjDot.restypes = None
ctl_lib.IjDot.argtypes = [
    ctl.ndpointer(dtype=np.single),
    ctl.ndpointer(dtype='b'), c_int, c_int,
    ctl.ndpointer(dtype=np.single)
]

# 神经元进程
def process_Neuron(niter: int, numNeuron: int, totalNeuron: int):

    # 初始化突触
    # Todo：WS 小世界连接
    WeightMask = np.random.choice([-1, 1, 0], size=(numNeuron, totalNeuron), p=[.2, .2, .6]).astype(np.int8)
    WeightRand = ((np.random.rand(numNeuron, totalNeuron)) * (1)).astype(np.single)
    WeightRand = np.multiply(WeightMask, WeightRand).astype(np.single)
    del WeightMask # 释放内存占用

    # 初始化神经元
    Vm = np.ones(numNeuron, dtype=np.single) * (-55)
    u = np.ones(numNeuron, dtype=np.single) * (-0)
    Ij = ((np.random.rand(numNeuron, totalNeuron)) * (5)).astype(np.single)
    Spike = np.zeros(numNeuron, dtype='b')
    SpikeAll = np.zeros(totalNeuron, dtype='b')

    # 神经元分类 E | I
    ClassNeuron = np.random.choice([2, 1], size=(numNeuron), p=[.8, .2]).astype(c_int)

    # 主进程
    if comm_rank == 0:
        import time

        # 绘图数组
        numPlot = 500
        picVe = np.ones(numPlot, dtype=np.single)
        picVi = np.ones(numPlot, dtype=np.single)
        picY = np.zeros((numPlot, totalNeuron), dtype=bool)
        
        # 打印发送辅助信息
        print("本程序为 WS 小世界 Izhikevich 神经元模型的 MPI 分布式仿真")
        print("Linux MPI 版本为：")
        os.system("mpirun --version | sed -n '1p'")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", totalNeuron)
        print("迭代次数为：", niter)

        # 记录仿真时长
        start = time.time()  

    # 神经元迭代
    for i in range(niter):
        # 计算神经元电位
        ctl_lib.SW(Vm, u, Ij, Spike, ClassNeuron, numNeuron)
        # 收集放电情况
        comm.Allgather(Spike, SpikeAll)
        # 计算突触电流
        ctl_lib.IjDot(WeightRand, SpikeAll, numNeuron, totalNeuron, Ij)

        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if (i < numPlot):
                picVe[i] = Vm[5]
                picVi[i] = Vm[6]
                picY[i] = SpikeAll
    
    # 主进程发送辅助信息
    if comm_rank == 0:
        print("运行时间为:", time.time() - start)

        # 绘制单个神经元实验结果验证仿真正确性
        import matplotlib
        import matplotlib.pyplot as plt
        
        matplotlib.use('Agg')
        plt.style.use('bmh')

        # SW Izhikevich Simulation results
        fig = plt.figure(dpi=300)
        x = np.linspace(0, numPlot, numPlot)
        # 膜电位
        ave = plt.subplot(2,2,1)
        ave.plot(x, picVe)
        ave.axes.xaxis.set_ticklabels([])
        ave.set_title("(b1)", x=1.05, y=0.8, size=10) # Membrane Potential
        ave.set_ylabel("Voltage/(mV)")
        ave.set_xlim(200,500)
        avi = plt.subplot(2,2,2)
        avi.plot(x, picVi)
        avi.axes.xaxis.set_ticklabels([])
        avi.set_title("(b1)", x=1.05, y=0.8, size=10) # Membrane Potential
        avi.set_ylabel("Voltage/(mV)")
        avi.set_xlim(200,500)
        # 放电栅格
        ay = plt.subplot(2,2,3)
        for i in range(numPlot):
            y = np.argwhere(picY[i] == 1)
            x = np.ones(len(y)) * i
            ay.scatter(x, y, c='black', s=0.5)
        ay.set_title("(b3)", x=1.05, y=0.8, size=10) # Firing Grid Map
        ay.set_ylabel("Neuron No.")
        ay.set_xlim(200,500)
        # 保存图片
        fig.tight_layout()
        fig.savefig(os.path.join(father_path, "SW.png"))

if __name__ == '__main__':
    # 初始化仿真参数
    if comm_rank == 0:
        numNeurons = 125
    else:
        numNeurons = 125
    totalNeurons = comm.allreduce(numNeurons)
    niters = 1000  # 迭代次数
    process_Neuron(niters, numNeurons, totalNeurons)
