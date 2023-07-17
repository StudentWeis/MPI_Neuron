#define dt 0.1

void izhikevich(double v, double u, double Ij, double a, double b, double *v1, double *u1)
{
    *v1 = 0.04 * v * v + 5 * v + 140 - u + Ij;
    *u1 = a * (b * v - u);
}

void IRungeKutta(double *v, double *u, double Ij, double a, double b, double c, double d)
{
    double v1, u1, v2, u2, v3, u3, v4, u4;
    izhikevich(*v, *u, Ij, a, b, &v1, &u1);
    izhikevich(*v + 0.5 * dt * v1, *u + 0.5 * dt * u1, Ij, a, b, &v2, &u2);
    izhikevich(*v + 0.5 * dt * v2, *u + 0.5 * dt * u2, Ij, a, b, &v3, &u3);
    izhikevich(*v + dt * v3, *u + dt * u3, Ij, a, b, &v4, &u4);
    *v = *v + 1 / 6 * dt * v1 + 1 / 3 * dt * v2 + 1 / 3 * dt * v3 + 1 / 6 * dt * v4;
    *u = *u + 1 / 6 * dt * u1 + 1 / 3 * dt * u2 + 1 / 3 * dt * u3 + 1 / 6 * dt * u4;
    // if (v >= 30)
    // {
    //     v = c;
    //     u = u + d;
    // }
}