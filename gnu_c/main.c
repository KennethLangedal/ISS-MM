#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "MM.h"

int main()
{
    const long long N = 2048, it = 1;
    float *A = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *C = (float *)aligned_alloc(32, N * N * sizeof(float));

    srand(0);

    for (int i = 0; i < N * N; i++)
        A[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N * N; i++)
        B[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;

    clock_t start = clock();

    for (int i = 0; i < it; i++)
        matmul(A, B, C, N);

    clock_t end = clock();

    double total_duration = (double)(end - start) / CLOCKS_PER_SEC;
    double average_duration = total_duration / it;
    double ops = N * N * N * 2LL; // N^3 multiplications and additions
    double flops = ops / average_duration;

    float cs = 0.0f;
    for (int i = 0; i < N * N; i++)
        cs += C[i];

    printf("%.5f %.5f %.5f\n", average_duration, flops / 1e9, cs);

    return 0;
}