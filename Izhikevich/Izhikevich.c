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
        izhikevich(Vm[i] + 0.05 * v1, u[i] + 0.05 * u1, Ij[i], a, b, &v2, &u2);
        izhikevich(Vm[i] + 0.05 * v2, u[i] + 0.05 * u2, Ij[i], a, b, &v3, &u3);
        izhikevich(Vm[i] + 0.1 * v3, u[i] + 0.1 * u3, Ij[i], a, b, &v4, &u4);
        Vm[i] += ((0.1 * v1 / 6) + (0.1 * v2 / 3) + (0.1 * v3 / 3) + (0.1 * v4 / 6));
        u[i] += ((0.1 * u1 / 6) + (0.1 * u2 / 3) + (0.1 * u3 / 3) + (0.1 * u4 / 6));
        if (Vm[i] >= 30)
        {
            Vm[i] = c;
            u[i] += d;
        }
    }
}