# MPI Neuron

使用 MPI 并行计算神经元。


# 虚拟环境初始化
conda create -n mpineuron python=3.10
conda install gcc_linux-64
pip install numpy matplotlib psutil mpi4py

# 使用

```sh
cd ~/MPI_Neuron
conda activate mpineuron
```

# LIF
export DISPLAY=:0
gcc -fPIC -shared LIF/LIF.c -o LIF/LIF.so -O3
mpiexec -n 16 python LIF/LIF_Nums.py
python LIF/NoMPI-LIF.py 

# HH
export DISPLAY=:0
g++ -fPIC -shared HH/HH.cpp -o HH/HH.so -O3
mpiexec --allow-run-as-root -n 6 python HH/HH.py

# Izhikevich
export DISPLAY=:0
mpiexec -n 4 python Izhikevich/Izhikevich.py