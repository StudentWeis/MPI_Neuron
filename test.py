import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('bmh')

fig = plt.figure(dpi=500, constrained_layout=False)
fig.tight_layout(pad=0.4,h_pad=5.0,w_pad=1.0)  

# LIF
x1=[10000, 20000, 30000, 40000, 60000, 68000, 70000]
y1=[0.07, 0.14, 0.27, 0.40, 0.83, 1, 1.1]

x2=[10000, 20000]
y2=[0.09, 0.21]

x3=[10000, 20000]
y3=[0.3, 0.7]

lif = plt.subplot(2, 2, 1)
lif.plot(x1,y1, marker='o',markersize=4,label='4-16')
lif.plot(x2,y2, marker='o',markersize=4,label='1-4')
lif.plot(x3,y3, marker='o',markersize=4,label='1-1')
lif.set_ylabel("Simulation Tims/(s)")
lif.set_xlabel("Number of Neurons")

# Izhikevich
x1=[10000, 20000, 30000, 40000, 42000, 45000]
y1=[0.15, 0.28, 0.46, 0.83, 1, 1.2]

x2=[10000, 20000]
y2=[0.25, 0.46]

x3=[10000, 20000]
y3=[0.91, 1.92]

izhikevich = plt.subplot(2, 2, 2)
izhikevich.plot(x1,y1, marker='o',markersize=4,label='4-16')
izhikevich.plot(x2,y2, marker='o',markersize=4,label='1-4')
izhikevich.plot(x3,y3, marker='o',markersize=4,label='1-1')
izhikevich.set_ylabel("Simulation Tims/(s)")
izhikevich.set_xlabel("Number of Neurons")

# HH
x1=[10000, 20000, 30000, 38000, 40000]
y1=[0.26, 0.40, 0.73, 1, 1.1]

x2=[10000, 20000]
y2=[0.48, 0.91]

x3=[10000, 20000]
y3=[1.61, 3.42]

HH = plt.subplot(2, 2, 3)
HH.plot(x1,y1, marker='o',markersize=4,label='4-16')
HH.plot(x2,y2, marker='o',markersize=4,label='1-4')
HH.plot(x3,y3, marker='o',markersize=4,label='1-1')
HH.set_ylabel("Simulation Tims/(s)")
HH.set_xlabel("Number of Neurons")

# 自适应调整间距
fig.tight_layout()
fig.savefig("./test.png")