#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "kernel.h"

int main()
{
    const int L2M = 12 * 10, L2N = 128, N = 384 * 200, it = 20;
    float *A = (float *)aligned_alloc(32, L2M * L2N * sizeof(float));
    float *B = (float *)aligned_alloc(32, L2N * N * sizeof(float));
    float *C = (float *)aligned_alloc(32, L2M * N * sizeof(float));

    srand(0);

    for (int i = 0; i < L2M * L2N; i++)
        A[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < L2N * N; i++)
        B[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    double best_duration = 1e10;

    for (int i = 0; i < it; i++)
    {
        for (int i = 0; i < L2M * N; i++)
            C[i] = 0.0f;

        struct timeval start, end;
        gettimeofday(&start, NULL);

        matmul(A, B, C, L2M, L2N, N);

        gettimeofday(&end, NULL);

        double duration = (double)(end.tv_usec - start.tv_usec) / 1000000.0 +
                          (double)(end.tv_sec - start.tv_sec);

        if (duration < best_duration)
            best_duration = duration;
    }

    // N^3 multiplications and additions
    double ops = (long long)L2M * (long long)L2N * (long long)N * 2LL;
    double flops = ops / best_duration;

    double cs = 0.0f;
    for (int i = 0; i < L2M * N; i++)
        cs += C[i];

    printf("%.5f %.5f %.5f\n", best_duration, flops / 1e9, cs);

    free(A);
    free(B);
    free(C);

    return 0;
}