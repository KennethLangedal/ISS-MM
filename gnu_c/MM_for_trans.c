#include "MM.h"

void matmul(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}