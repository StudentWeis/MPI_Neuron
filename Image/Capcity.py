# 交错式内存图
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
matplotlib.use('Agg')
# plt.style.use('bmh')

fig = plt.figure(dpi=500)

x = np.arange(0, 9, 1)
ay = np.array([0.1353, 0.5083, 0.8084, 0.4152, 0.4152, 0.4152, 0.4152, 0.4152, 0.4152])
by = np.array([0.52, 2.1, 3.2, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62]) - ay
cy = np.array([0.1353, 0.5083, 0.8084, 1.2152, 0.8152, 1.6152, 1.2152, 2, 1.65]) -by -ay
y = np.vstack([ay, by, cy])

ax = plt.subplot(1,1,1)
ax.stackplot(x, y)
ax.set(xlim=(0, 8), xticks=np.arange(1, 9),
       ylim=(0, 6), yticks=np.arange(1, 6))

fig.savefig('test.jpg')