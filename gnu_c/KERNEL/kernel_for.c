#include "kernel.h"

void matmul(const float *A, const float *B, float *C, int L2M, int L2N, int N)
{
    for (int i = 0; i < L2M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < L2N; k++)
            {
                C[i * N + j] += A[i * L2N + k] * B[k * N + j];
            }
        }
    }
}