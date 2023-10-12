#include <iostream>
using namespace std;

// 仿真步长
#define dt 0.8

void Izhikevich(float Vmi, float ui, float Iji, float a, float b, float *vt,
                float *ut) {
  *vt = 0.04 * Vmi * Vmi + 5 * Vmi + 140 - ui + Iji;
  *ut = a * (b * Vmi - ui);
}

// 小世界
float v1, u1, v2, u2, v3, u3, v4, u4;
extern "C" void SW(float *Vm, float *u, float *Ij, char *Spike, int *Class,
                   int numNeu) {
  float a, b, c, d;
  for (int i = 0; i < numNeu; i++) {
    switch (Class[i]) {
    case 0: // 颤动型
      a = 0.02;
      b = 0.2;
      c = -50.0;
      d = 2;
      break;
    case 1: // 兴奋型
      a = 0.02;
      b = 0.2;
      c = -65.0;
      d = -8;
      break;
    case 2: // 抑制型
      a = 0.02;
      b = 0.25;
      c = -65.0;
      d = 2;
      break;
    }
    if (Spike[i]) {
      Vm[i] = c;
      Spike[i] = 0;
    }

    Izhikevich(Vm[i], u[i], Ij[i], a, b, &v1, &u1);
    Izhikevich(Vm[i] + 0.5 * dt * v1, u[i] + 0.5 * dt * u1, Ij[i], a, b, &v2,
               &u2);
    Izhikevich(Vm[i] + 0.5 * dt * v2, u[i] + 0.5 * dt * u2, Ij[i], a, b, &v3,
               &u3);
    Izhikevich(Vm[i] + dt * v3, u[i] + dt * u3, Ij[i], a, b, &v4, &u4);
    Vm[i] += ((dt * v1 / 6) + (dt * v2 / 3) + (dt * v3 / 3) + (dt * v4 / 6));
    u[i] += ((dt * u1 / 6) + (dt * u2 / 3) + (dt * u3 / 3) + (dt * u4 / 6));
    if (Vm[i] >= 30) {
      Vm[i] = 30;
      u[i] += d;
      Spike[i] = 1;
    }
  }
}

// 计算突触电流
extern "C" void IjDot(float *Weight, char *SpikeAll, int numNeu,
                      int totalNeuron, float *Ij) {
  for (int i = 0; i < numNeu; i++) {
    Ij[i] = 5;
  }
  for (int i = 0; i < numNeu; i++) {
    for (int j = 0; SpikeAll[j] > 0 && j < totalNeuron; j++) {
      Ij[i] += Weight[i * totalNeuron + j];
    }
  }
}