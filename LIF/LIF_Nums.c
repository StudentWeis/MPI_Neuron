#define tref 3 // refractory period in msec
#define C 0.2  // capacitance in nF
#define R 100  // resitance in megaohm
#define taum (R * C)

// 考虑灭火期
void lifPI(float *Vm, char *Spike, int numNeu, float *Ij, unsigned char *period)
{
    for (int i = 0; i < numNeu; i++)
    {
        if (period[i] == 0)
        {
            if (Vm[i] > -60 && Vm[i] < 0)
            {
                Vm[i] = 0;
                Spike[i] = 1;
                continue;
            }
            else if (Vm[i] >= 0)
            {
                Vm[i] = -70;
                period[i] = tref;
                // 这里不能加 continue，否则无法造成随机性
            }
            Vm[i] += ((1 / taum) * (-70 - Vm[i] + R * Ij[i]));
        }
        else
        {
            period[i]--;
        }
    }
}

// 计算突触电流
void IjDot(float *Weight, char *SpikeAll, int numNeu, int totalNeuron, float *Ij)
{
    for (int i = 0; i < numNeu; i++)
    {
        Ij[i] = 0.15;
    }
    for (int i = 0; i < numNeu; i++)
    {
        for (int j = 0; SpikeAll[j] > 0 && j < totalNeuron; j++)
        {
            Ij[i] += Weight[i * totalNeuron + j];
        }
    }
}