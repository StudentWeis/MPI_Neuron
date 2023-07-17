import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

a = 0.05  # 恢复变量的时间尺度，越小，恢复越慢
b = 0.2  # 恢复变量依赖膜电位的阈值下随机波动的敏感度
c = -50.0  # 膜电位复位值
d = 5  # 恢复变量复位值


def izhikevich(v, u, I):  # 神经元模型
    v1 = 0.04*v*v+5*v+140-u+I
    u1 = a*(b*v-u)
    return v1, u1


def rungeKutta(v, u, t, I):  # runge-kutta求解
    v1, u1 = izhikevich(v, u, I)
    v2, u2 = izhikevich(v+0.5*t*v1, u+0.5*t*u1, I)
    v3, u3 = izhikevich(v+0.5*t*v2, u+0.5*t*u2, I)
    v4, u4 = izhikevich(v+t*v3, u+t*u3, I)
    vf = v+1/6*t*v1+1/3*t*v2+1/3*t*v3+1/6*t*v4
    uf = u+1/6*t*u1+1/3*t*u2+1/3*t*u3+1/6*t*u4
    return vf, uf


def main():
    v = -65  # 初始膜电位
    u = -0  # 初始恢复量
    t = 0.1  # 时间步长
    I = 5  # 输入电流
    vs = []  # 记录v
    us = []  # 记录u
    time = []  # 时间戳
    for i in np.arange(0, 1000, t):  # 模拟时长(ms)
        time.append(i)
        v, u = rungeKutta(v, u, t, I)
        if v >= 30:  # 发射脉冲
            vs.append(30)
            v = c
            u = u+d
        else:
            vs.append(v)
        us.append(u)
    plt.plot(time, vs)
    plt.savefig("test.png")


main()
