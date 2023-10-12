import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
plt.style.use('bmh')

fig = plt.figure(dpi=500)

xlabels = ['7B-Q4', '7B-Q5', '7B-Q8', '13B-Q2', '13B-Q4', '13B-Q5']
x = np.arange(len(xlabels))

# 柱状图
axs = plt.subplot()
width = 0.25
ModelSize = [3.79, 4.63, 5.51, 7.16, 6.8, 10.68]
ModelRAM = [6.29, 7.13, 7.89, 9.66, 9.82, 13.18]
rects1 = axs.bar(x - 0.5 * width, ModelSize, width, label='Size(Left)', color='grey')
rects2 = axs.bar(x + 0.5 * width, ModelRAM, width, label='RAM(Left)',color='brown')
axs.legend()
axs.set_ylabel('Model Size (GB)')

# 折线图
axt = axs.twinx()
xm4 = np.arange(3)
macTime = [55, 76, 89, 72, 103, 148]
m8Time = [183.47, 248.18, 260.38, 239.68, 374.64, 475.43]
m4Time = [206.67, 269.58, 277.82]

axt.plot(x, macTime, marker='o', label='MacBook(Right)', color='orange')
axt.plot(x, m8Time, marker='o', label='MEDC-8(Right)', color='deepskyblue')
axt.plot(xm4, m4Time, marker='o', label='MEDC-4(Right)', color='green')
axt.set_xticks(x, xlabels)

axt.set_ylabel('Speed (ms/Token)', rotation=-90, labelpad=15)
axt.legend(bbox_to_anchor=(0.34, 0.88))

axs.set_xlabel('Type of Models')

fig.tight_layout()
fig.savefig('img.png')