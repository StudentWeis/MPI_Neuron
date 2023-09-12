# 交错式内存初始化
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
matplotlib.use('Agg')
plt.style.use('bmh')

fig = plt.figure(dpi=500, figsize=(7, 3.25))

# 内存初始化图
mem = plt.subplot(1,2,1)

x = np.arange(0, 10, 1)
ay = np.array([0.1353, 0.5083, 0.8084, 0.4152, 0.4152, 0.4152, 0.4152, 0.4152, 0.4152, 0.4152])
by = np.array([0.52, 2.1, 3.2, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62]) - ay
cy = np.array([0.1353, 0.5083, 0.8084, 1.2152, 0.8152, 1.6152, 1.2152, 2, 1.62, 1.62]) -by -ay

mem.stackplot(x, ay, by, cy, labels=['Single', 'Interlace', 'Overlap'])
mem.set_xlim(0,9)
mem.set_ylabel('Storage/(GB)')
mem.set_xlabel('Startup Step')
mem.set_title('(a)', loc='right')
mem.set_title('Memory')
mem.legend()
mem.grid(False)

# 容量对比图
cap = plt.subplot(1, 2, 2)
xlabels = ['LIF', 'Izhikevich', 'HH']
x = np.arange(len(xlabels))
y1 = [24000, 5500, 5250]
y2 = [22000, 5250, 4900]
y3 = [11500, 2650 ,2500]
width = 0.25
rects1 = cap.bar(x - width, y1, width, label='Single')
rects2 = cap.bar(x, y2, width, label='Interlace')
rects3 = cap.bar(x + width, y3, width, label='Overlap')
cap.set_ylabel('Number of Neurons')
cap.set_xlabel('Type of Neurons')
cap.set_title('(b)', loc='right')
cap.set_title('Capacity')
cap.set_ylim(0,28000)
cap.set_xticks(x, xlabels)
cap.ticklabel_format(useMathText=True, style='sci', scilimits=(-1,2), axis='y')
cap.legend()



# 保存图片
fig.tight_layout()
fig.savefig('Interleaved Memory Initialization Capacity.png')