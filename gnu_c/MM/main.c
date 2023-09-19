#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "MM.h"

int main()
{
    const long long N = 1 << 11, it = 3;
    float *A = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *Bt = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *C = (float *)aligned_alloc(32, N * N * sizeof(float));

    srand(0);

    for (int i = 0; i < N * N; i++)
        A[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N * N; i++)
        B[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;

    for (int i = 0; i < N * N; i++)
        Bt[i] = 0.0f;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < it; i++)
        matmul(A, B, C, Bt, N);

    gettimeofday(&end, NULL);

    double total_duration = (double)(end.tv_usec - start.tv_usec) / 1000000.0 +
                            (double)(end.tv_sec - start.tv_sec);
    double average_duration = total_duration / it;
    double ops = N * N * N * 2LL; // N^3 multiplications and additions
    double flops = ops / average_duration;

    double cs = 0.0f;
    for (int i = 0; i < N * N; i++)
        cs += C[i];

    printf("%.5f %.5f %.5f\n", average_duration, flops / 1e9, cs);

    return 0;
}