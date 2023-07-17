# 本程序为 LIF 神经元的 MPI 分布式仿真
# Num of Neurons is: 10000
# Num of Niter is 1000
# Time of Process is: 0.7344927787780762

import time
import numpy as np
from mpi4py import MPI
import matplotlib

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


class LIFNerons():
    # LIF 神经元最小集群定义

    numNeuron = 5000  # 最小集群神经元数量

    def __init__(self):
        self.Vm = np.ones(self.numNeuron) * (-70)

    def step(self):
        # LIF 神经元单步计算
        # TODO: 输入一个放电数组，先根据放电数组情况计算出输入电流
        Spike = np.zeros(self.numNeuron, bool)  # 放电序列，不需要保存
        for i, VmC in enumerate(self.Vm):
            VmC += 0.05 * (-55 - VmC)
            if (VmC > -60):
                VmC = -70
                Spike[i] = 1
            self.Vm[i] = VmC
        # 返回放电情况
        return Spike


def process_Neuron(niter):
    # 每个 MPI 进程初始化神经元
    NeuC = LIFNerons()
    Spike = np.empty(LIFNerons.numNeuron, dtype='b')
    SpikeAll = np.empty(comm_size * LIFNerons.numNeuron, dtype='b')

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Num of Neurons is:", comm_size * LIFNerons.numNeuron)
        print("Num of Niter is", niter)
        start = time.time()

    # 每个进程分别迭代计算，然后全收集
    for _ in range(niter):
        Spike = NeuC.step()
        comm.Allgather(Spike, SpikeAll)
        # TODO: 绘制单个神经元的膜电位曲线验证仿真正确性

    # 主进程发送辅助信息
    if comm_rank == 0:
        print("Time of Process is:", time.time() - start)


# 主程序
if __name__ == '__main__':
    if comm_rank == 0:
        print("本程序为 LIF 神经元的 MPI 分布式仿真")
    process_Neuron(1000)
