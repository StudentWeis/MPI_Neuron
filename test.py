import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
matplotlib.use('Agg')
plt.style.use('bmh')

fig = plt.figure(dpi=500)

# LIF
lif = plt.subplot(2, 2, 1)
x1=[4000, 10000, 20000, 30000, 40000, 60000, 68000, 70000]
y1=[0.02, 0.04, 0.13, 0.27, 0.40, 0.83, 1, 1.1]

x2=[4000, 10000, 15000, 22000]
y2=[0.05, 0.11, 0.18, 0.29]

x3=[4000, 10000, 20000, 24000]
y3=[0.14, 0.3, 0.7, 0.9]

# STM32
x4=[4000, 8000, 10000, 12000, 16000]
y4=[0.5, 0.76, 1.0, 1.22, 1.76]
lif.plot(x1,y1, marker='o',markersize=2,linewidth=1.5,label='4-16')
lif.plot(x2,y2, marker='o',markersize=2,linewidth=1.5,label='1-4')
lif.plot(x3,y3, marker='o',markersize=2,linewidth=1.5,label='1-1')
lif.plot(x4,y4, marker='o',markersize=2,linewidth=1.5,label='STM32')
lif.set_ylabel("Simulation Tims/(s)")
lif.set_xlabel("Number of Neurons")
lif.set_ylim(0,2)
lif.set_title('(a)', loc='right')
lif.set_title('LIF')
lif.legend(ncol=2, prop = {'size':8})
lif.ticklabel_format(useMathText=True, style='sci', scilimits=(-1,2), axis='x')

# Izhikevich
izhikevich = plt.subplot(2, 2, 2)
x1=[880, 5000, 10000, 15000, 20000, 21000, 22500]
y1=[0.01, 0.15, 0.28, 0.46, 0.83, 1, 1.2]

x2=[880, 2500, 3500, 5250]
y2=[0.03, 0.25, 0.32, 0.46]

x3=[880, 2500, 3500, 5500]
y3=[0.08, 0.91, 1.52, 1.92]

# STM32
x4=[880, 1200, 1600]
y4=[1, 1.38, 1.8]

izhikevich.plot(x1,y1, marker='o',markersize=2,linewidth=1.5,label='4-16')
izhikevich.plot(x2,y2, marker='o',markersize=2,linewidth=1.5,label='1-4')
izhikevich.plot(x3,y3, marker='o',markersize=2,linewidth=1.5,label='1-1')
izhikevich.plot(x4,y4, marker='o',markersize=2,linewidth=1.5,label='STM32')
izhikevich.set_ylabel("Simulation Tims/(s)")
izhikevich.set_xlabel("Number of Neurons")
izhikevich.set_title('(b)', loc='right')
izhikevich.set_title('Izhikevich')
izhikevich.legend(ncol=2, prop = {'size':8})
izhikevich.ticklabel_format(useMathText=True, style='sci', scilimits=(-1,2), axis='x')

# HH
hh = plt.subplot(2, 2, 3)
x1=[88, 5000, 10000, 15000, 19000, 20000]
y1=[0.01, 0.26, 0.40, 0.73, 1.00, 1.1]

x2=[88, 2500, 3500, 4900]
y2=[0.01, 0.48, 0.78, 0.91]

x3=[88, 2500, 3500, 5250]
y3=[0.01, 1.61, 2.45, 3.42]

# STM32
x4=[88, 120, 160]
y4=[1, 1.28, 1.7]

hh.plot(x1,y1, marker='o',markersize=2,linewidth=1.5,label='4-16')
hh.plot(x2,y2, marker='o',markersize=2,linewidth=1.5,label='1-4')
hh.plot(x3,y3, marker='o',markersize=2,linewidth=1.5,label='1-1')
hh.plot(x4,y4, marker='o',markersize=2,linewidth=1.5,label='STM32')
hh.set_ylabel("Simulation Tims/(s)")
hh.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
hh.set_xlabel("Number of Neurons")
hh.set_title('(c)', loc='right')
hh.set_title('HH')
hh.legend(ncol=2, prop = {'size':8})
hh.ticklabel_format(useMathText=True, style='sci', scilimits=(-1,2), axis='x')

# Cap
cap = plt.subplot(2, 2, 4)
xlabels = ['LIF', 'Izhikevich', 'HH']
y1 = [70000, 22500, 20000]
y2 = [22000, 5250, 4900]
y3 = [24000, 5500 ,5250]
y4 = [16000, 1600, 160]

x = np.arange(len(xlabels))
width = 0.2
rects1 = cap.bar(x - 3/2 * width, y1, width, label='4-16')
rects2 = cap.bar(x - 1/2 * width, y2, width, label='1-4')
rects3 = cap.bar(x + 1/2 * width, y3, width, label='1-1')
rects3 = cap.bar(x + 3/2 * width, y4, width, label='STM32')
cap.set_ylabel('Number of Neurons')
cap.set_xlabel('Type of Neurons')
cap.set_xticks(x, xlabels)
cap.set_ylim(0, 80000)
cap.ticklabel_format(useMathText=True, style='sci', scilimits=(-1,2), axis='y')
cap.set_title('(d)', loc='right')
cap.legend(ncol=2, prop = {'size':8})
cap.set_title('Capacity')

# 自适应调整间距
fig.tight_layout()
fig.savefig("./test.png")