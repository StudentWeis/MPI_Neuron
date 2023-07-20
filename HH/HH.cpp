#include <math.h>
#include <stdio.h>

#define ENa 115.0 // Na+离子的平衡电位
#define EK -12.0  // K+离子的平衡电位
#define El 10.6   // 其他离子流的平衡电位
#define gNa 120.0 // Na+通道的最大电导
#define gK 36.0   // K+通道的最大电导
#define gl 0.3    // 其他离子流的最大电导
#define dt 0.01   // ms
#define Vr -70

extern "C" void HH(double *Vm, int numNeu, double *n, double *m, double *h)
{
    for (int i = 0; i < numNeu; i++)
    {
        double alpha_n = (0.1 - 0.01 * Vm[i]) / (exp(1 - 0.1 * Vm[i]) - 1);
        double alpha_m = (2.5 - 0.1 * Vm[i]) / (exp(2.5 - 0.1 * Vm[i]) - 1);
        double alpha_h = 0.07 * exp(-1 * Vm[i] / 20.0);
        double beta_n = 0.125 * exp(-1 * Vm[i] / 80.0);
        double beta_m = 4.0 * exp(-1 * Vm[i] / 18.0);
        double beta_h = 1 / (exp(3 - 0.1 * Vm[i]) + 1);

        double I_Na = gNa * pow(m[i], 3) * h[i] * (Vm[i] - ENa);
        double I_K = gK * pow(n[i], 4) * (Vm[i] - EK);
        double I_L = gl * (Vm[i] - El);
        double dv = 15.0 - (I_Na + I_K + I_L);
        Vm[i] = Vm[i] + dv * dt;

        m[i] = m[i] + (alpha_m * (1 - m[i]) - beta_m * m[i]) * dt;
        n[i] = n[i] + (alpha_n * (1 - n[i]) - beta_n * n[i]) * dt;
        h[i] = h[i] + (alpha_h * (1 - h[i]) - beta_h * h[i]) * dt;
    }
}