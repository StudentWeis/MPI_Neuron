#include <stdio.h>

void izhikevich(float Vmi, float ui, float Iji, float a, float b, float *vt, float *ut)
{
    vt[0] = 0.04 * Vmi * Vmi + 5 * Vmi + 140 - ui + Iji;
    ut[0] = a * (b * Vmi - ui);
}
float v1, u1, v2, u2, v3, u3, v4, u4;
void rungeKutta(float *Vm, float *u, float *Ij, float a, float b, float c, float d, int numNeu)
{
    for (int i = 0; i < numNeu; i++)
    {
        izhikevich(Vm[i], u[i], Ij[i], a, b, &v1, &u1);
        izhikevich(Vm[i] + 0.5 * v1, *u + 0, Ij[i], a, b, &v2, &u2);
        izhikevich(Vm[i] + 0.5 * v2, *u + 0.5 * u2, Ij[i], a, b, &v3, &u3);
        izhikevich(Vm[i] + v3, *u + u3, Ij[i], a, b, &v4, &u4);

        printf("first: v1=%f, v2=%f, v3=%f, v4=%f\n, ", v1, v2, v3, v4);
        Vm[i] += (1 / 6) * v1 + (1 / 3) * v2 + (1 / 3) * v3 + (1 / 6) * v4;
        u[i] += (1 / 6) * u1 + (1 / 3) * u2 + (1 / 3) * u3 + (1 / 6) * u4;
        // if (Vm[i] >= 30)
        // {
        //     Vm[i] = c;
        //     u[i] = u[i] + d;
        // }
    }
}