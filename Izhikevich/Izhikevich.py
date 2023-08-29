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
    c_float, c_float, c_float, c_float, c_int
]

def process_Neuron(niter: int, numNeuron: int):
    # a = 0.05  # 恢复变量的时间尺度，越小，恢复越慢
    # b = 0.2  # 恢复变量依赖膜电位的阈值下随机波动的敏感度
    # c = -50.0  # 膜电位复位值
    # d = 5  # 恢复变量复位值

    a = 0.02  # 恢复变量的时间尺度，越小，恢复越慢
    b = 0.2  # 恢复变量依赖膜电位的阈值下随机波动的敏感度
    c = -50.0  # 膜电位复位值
    d = 2  # 恢复变量复位值
    Vm = np.ones(numNeuron, dtype=np.single) * (-55)
    u = np.ones(numNeuron, dtype=np.single) * (-0)
    Ij = np.ones(numNeuron, dtype=np.single) * (5)

    # 主进程
    if comm_rank == 0:
        import time

        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        numPlot = 10000
        # 打印发送辅助信息
        print("本程序为 Izhikevich 神经元的 MPI 分布式仿真")
        print("Linux MPI 版本为")
        print("使用 C 动态库进行加速计算")
        print("使用 Matplotlib 绘图")
        print("神经元数量为：", numNeuron)
        print("迭代次数为：", niter)
        # 绘图数组
        picU = np.ones(numPlot, dtype=np.single)

    for i in range(niter):
        ctl_lib.rungeKutta(Vm, u, Ij, a, b, c, d, numNeuron)
        # 记录单个神经元的膜电位数据
        if comm_rank == 0:
            if (i < numPlot):
                picU[i] = Vm[5]
    
        # 主进程发送辅助信息
    if comm_rank == 0:
        # 绘制单个神经元的膜电位曲线验证仿真正确性
        x = np.linspace(0, numPlot, numPlot)
        figU, ax = plt.subplots()
        ax.plot(x, picU)
        figU.savefig("Izhikevich 膜电位.png")

if __name__ == '__main__':
    # 初始化仿真参数
    niters = 10000  # 迭代次数
    process_Neuron(niters, 10)
