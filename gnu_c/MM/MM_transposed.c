#include "MM.h"

__attribute__((optimize("no-tree-vectorize")))
void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}