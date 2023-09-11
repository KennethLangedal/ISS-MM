#include "MM.h"
#include <immintrin.h>

void matmul(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            __m256 a = _mm256_broadcast_ss(A + i * N + k);
            for (int j = 0; j < N; j += 8)
            {
                __m256 b = _mm256_load_ps(B + k * N + j);
                __m256 c = _mm256_load_ps(C + i * N + j);

                c = _mm256_fmadd_ps(a, b, c);

                _mm256_store_ps(C + i * N + j, c);
            }
        }
    }
}