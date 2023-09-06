import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
plt.style.use('bmh')

fig = plt.figure(dpi=500, constrained_layout=False)
fig.tight_layout(pad=0.4,h_pad=5.0,w_pad=1.0)  

# LIF
lif = plt.subplot(2, 2, 1)
x1=[4000, 10000, 20000, 30000, 40000, 60000, 68000, 70000]
y1=[0.02, 0.04, 0.13, 0.27, 0.40, 0.83, 1, 1.1]

x2=[4000, 10000, 15000, 22000]
y2=[0.05, 0.11, 0.18, 0.29]

x3=[4000, 10000, 20000, 24000]
y3=[0.14, 0.3, 0.7, 0.9]

x4=[4000, 8000, 10000, 12000, 16000]
y4=[0.5, 0.76, 1.0, 1.22, 1.76]

lif.plot(x1,y1, marker='o',markersize=3,label='4-16')
lif.plot(x2,y2, marker='o',markersize=3,label='1-4')
lif.plot(x3,y3, marker='o',markersize=3,label='1-1')
lif.plot(x4,y4, marker='o',markersize=3,label='STM32')
lif.scatter(68000,1.0,s=20,color='r', zorder=10) 
lif.set_ylabel("Simulation Tims/(s)")
lif.set_xlabel("Number of Neurons")
lif.set_title('(a)', loc='right')

# Izhikevich
izhikevich = plt.subplot(2, 2, 2)
x1=[10000, 20000, 30000, 40000, 42000, 45000]
y1=[0.15, 0.28, 0.46, 0.83, 1, 1.2]

x2=[10000, 21000]
y2=[0.25, 0.46]

x3=[10000, 21500]
y3=[0.91, 1.92]

izhikevich.plot(x1,y1, marker='o',markersize=4,label='4-16')
izhikevich.plot(x2,y2, marker='o',markersize=4,label='1-4')
izhikevich.plot(x3,y3, marker='o',markersize=4,label='1-1')
izhikevich.set_ylabel("Simulation Tims/(s)")
izhikevich.set_xlabel("Number of Neurons")
izhikevich.set_title('(b)', loc='right')

# HH
hh = plt.subplot(2, 2, 3)
x1=[10000, 20000, 30000, 38000, 40000]
y1=[0.26, 0.40, 0.73, 1.00, 1.1]

x2=[10000, 20000]
y2=[0.48, 0.91]

x3=[10000, 21000]
y3=[1.61, 3.42]

hh.plot(x1,y1, marker='o',markersize=4,label='4-16')
hh.plot(x2,y2, marker='o',markersize=4,label='1-4')
hh.plot(x3,y3, marker='o',markersize=4,label='1-1')
hh.set_ylabel("Simulation Tims/(s)")
hh.set_xlabel("Number of Neurons")
hh.set_title('(c)', loc='right')

# Cap
cap = plt.subplot(2, 2, 4)
labels = ['LIF', 'Izhikevich', 'HH']
y1 = [70000, 45000, 40000]
y2 = [22000, 21000, 20000]
y3 = [24000, 21500 ,21000]

x = np.arange(len(labels))
width = 0.3
rects1 = cap.bar(x - width, y1, width, label='4-16')
rects2 = cap.bar(x, y2, width, label='1-4')
rects3 = cap.bar(x + width, y3, width, label='1-1')
cap.set_ylabel('Number of Neurons')
cap.set_xlabel('Type of Neurons')
cap.set_xticks(x, labels)
cap.set_ylim(0, 100000)
cap.bar_label(rects1, padding=3)
cap.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
cap.set_title('(d)', loc='right')
cap.legend()

# 自适应调整间距
fig.tight_layout()
fig.savefig("./test.png")