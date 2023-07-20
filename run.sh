cd ~/MPI_Neuron

# 虚拟环境
conda activate mpineuron
pip install numpy matplotlib psutil

# LIF
gcc -fPIC -shared LIF/LIF.c -o LIF/LIF.so -O3
mpiexec -n 4 python LIF/LIF_Nums.py

# HH
export DISPLAY=:0
g++ -fPIC -shared HH/HH.cpp -o HH/HH.so -O3
mpiexec --allow-run-as-root -n 6 python HH/HH.py