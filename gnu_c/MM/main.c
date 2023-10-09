#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "MM.h"

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("Give arguments M, N, K, and it\n");
        return 0;
    }
    const size_t M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), it = atoi(argv[4]);
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
    // N^3 multiplications and additions
    double ops = M * N * K * 2LL;

    for (int i = 0; i < it; i++)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        matmul(A, B, C, M, N, K);

        gettimeofday(&end, NULL);

        double duration = (double)(end.tv_usec - start.tv_usec) / 1e6 +
                          (double)(end.tv_sec - start.tv_sec);

        if (duration < best_duration)
            best_duration = duration;

        double flops = ops / duration;
        printf("Iteration %d: %.5f %.5f\n", i, duration, flops / 1e9);
    }

    double flops = ops / best_duration;
    double cs = 0.0f;
    for (int i = 0; i < M * N; i++)
        cs += C[i];

    printf("Best: %.5f %.5f %.5f\n", best_duration, flops / 1e9, cs);

    free(A);
    free(B);
    free(C);

    return 0;
}