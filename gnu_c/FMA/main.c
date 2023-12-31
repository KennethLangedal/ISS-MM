#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "FMA.h"

int main()
{
    srand(0);

    int N = 1 << 26;

    float *A = (float *)aligned_alloc(32, N * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * sizeof(float));
    float *C = (float *)aligned_alloc(32, N * sizeof(float));

    for (int i = 0; i < N; i++)
        A[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N; i++)
        B[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N; i++)
        C[i] = 0.0f;

    for (int k = 256; k < N; k <<= 1)
    {
        for (int n = k; n < (k << 1); n += k / 4)
        {
            long long it = (N / n) * 10;

            for (int i = 0; i < n; i++)
                C[i] = 0.0f;

            struct timeval start, end;
            gettimeofday(&start, NULL);

            FMA(A, B, C, n, it);

            gettimeofday(&end, NULL);

            double total_duration = (double)(end.tv_usec - start.tv_usec) / 1e6 +
                                    (double)(end.tv_sec - start.tv_sec);
            double average_duration = total_duration / it;
            double ops = n * ops_per_element;
            double flops = ops / average_duration;

            float cs = 0.0f;
            for (int i = 0; i < n; i++)
                cs += C[i];

            printf("%d %.5f\n", n, flops / 1e9);
        }
    }

    return 0;
}