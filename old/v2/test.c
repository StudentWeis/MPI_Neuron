#define tref 0 // refractory period in msec
#define C 0.2       // capacitance in nF
#define R 100       // resitance in megaohm
#define taum (R * C)

void lifPI(double * Vm, char * Spike, int numNeu, double * Ij, unsigned char * period) {
    for(int i = 0; i < numNeu; i++) {
        if(period[i] == 0) {
            if(Vm[i] > -60) {
                Vm[i] = -70;
                Spike[i] = 1;
                period[i] = 3;
            }
            Vm[i] += (0.05 * (-70 - Vm[i] + R * Ij[i]));
        } else {
            period[i]--;
        }
    }
}

// 计算突触电流
void IjDot(double * Weight, char * SpikeAll, int numNeu, int numProc, double * Ij) {
    int temp = numNeu * numProc;
    for(int i = 0; i < numNeu; i++) {
        Ij[i] = 0.25;
    }
    for(int i = 0; i < numNeu; i++) {
        for(int j = 0; SpikeAll[j] > 0 && j < temp; j++) {
            Ij[i] += Weight[i * temp + j];
        }
    }
}