#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "MM.h"

int main()
{
    const int M = 132 * 32, N = 256 * 16, K = 256 * 16, it = 3;
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    srand(0);

    for (int i = 0; i < M * K; i++)
        A[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < K * N; i++)
        B[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    double best_duration = 1e10;

    for (int i = 0; i < it; i++)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        matmul(A, B, C, M, N, K);

        gettimeofday(&end, NULL);

        double duration = (double)(end.tv_usec - start.tv_usec) / 1000000.0 +
                          (double)(end.tv_sec - start.tv_sec);

        if (duration < best_duration)
            best_duration = duration;
    }

    // N^3 multiplications and additions
    double ops = (long long)M * (long long)N * (long long)K * 2LL;
    double flops = ops / best_duration;

    double cs = 0.0f;
    for (int i = 0; i < M * N; i++)
        cs += C[i];

    printf("%.5f %.5f %.5f\n", best_duration, flops / 1e9, cs);

    free(A);
    free(B);
    free(C);

    return 0;
}