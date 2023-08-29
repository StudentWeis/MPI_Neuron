void izhikevich(float Vmi, float ui, float Iji, float a, float b, int numNeu, float *vt, float *ut)
{
    for (int i = 0; i < numNeu; i++)
    {
        *vt = 0.04 * Vmi * Vmi + 5 * Vmi + 140 - ui + Iji;
        *ut = a * (b * Vmi - ui);
    }
}

void rungeKutta(float *Vm, float *u, float *Ij, float a, float b, float c, float d, int numNeu)
{
    for (int i = 0; i < numNeu; i++)
    {
        float v1, u1, v2, u2, v3, u3, v4, u4;
        izhikevich(Vm[i], u[i], Ij[i], a, b, numNeu, &v1, &u1);
        izhikevich(Vm[i] + 0.5 * v1, *u + 0.5 * u1, Ij[i], a, b, numNeu, &v2, &u2);
        izhikevich(Vm[i] + 0.5 * v2, *u + 0.5 * u2, Ij[i], a, b, numNeu, &v3, &u3);
        izhikevich(Vm[i] + v3, *u + u3, Ij[i], a, b, numNeu, &v4, &u4);

        Vm[i] = Vm[i] + (1 / 6) * v1 + (1 / 3) * v2 + (1 / 3) * v3 + (1 / 6) * v4;
        u[i] = u[i] + (1 / 6) * u1 + (1 / 3) * u2 + (1 / 3) * u3 + (1 / 6) * u4;
        // if (Vm[i] >= 30)
        // {
        //     Vm[i] = c;
        //     u[i] = u[i] + d;
        // }
    }
}