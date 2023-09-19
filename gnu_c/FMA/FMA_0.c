#include "FMA.h"

void FMA(const float *a, const float *b, float *c, int n, int it)
{
    for (int t = 0; t < it; t++)
        for (int i = 0; i < n; i++)
            c[i] = a[i] * b[i] + c[i];
}