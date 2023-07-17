cd ~/mpi/v3

# LIF
export DISPLAY=:0
gcc -fPIC -shared LIF/LIF.c -o LIF/LIF.so -O3
mpiexec --allow-run-as-root -n 6 python LIF/LIF.py

# HH
export DISPLAY=:0
g++ -fPIC -shared HH/HH.cpp -o HH/HH.so -O3
mpiexec --allow-run-as-root -n 6 python HH/HH.py