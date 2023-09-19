#include "MM.h"

#include <immintrin.h>

typedef float v8sf __attribute__((vector_size(32)));

static inline void matmul_4_16_kernel(const float *A, const float *B, float *C, int N, int ci, int cj)
{
    v8sf c00 = _mm256_setzero_ps();
    v8sf c01 = _mm256_setzero_ps();

    v8sf c10 = _mm256_setzero_ps();
    v8sf c11 = _mm256_setzero_ps();

    v8sf c20 = _mm256_setzero_ps();
    v8sf c21 = _mm256_setzero_ps();

    v8sf c30 = _mm256_setzero_ps();
    v8sf c31 = _mm256_setzero_ps();

    for (int i = 0; i < N; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * N + cj);
        v8sf b1 = _mm256_load_ps(B + i * N + cj + 8);

        v8sf a0 = _mm256_broadcast_ss(A + (ci + 0) * N + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + (ci + 1) * N + i);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + (ci + 2) * N + i);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + (ci + 3) * N + i);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);
    }

    _mm256_store_ps(C + (ci + 0) * N + cj + 0, c00);
    _mm256_store_ps(C + (ci + 0) * N + cj + 8, c01);

    _mm256_store_ps(C + (ci + 1) * N + cj + 0, c10);
    _mm256_store_ps(C + (ci + 1) * N + cj + 8, c11);

    _mm256_store_ps(C + (ci + 2) * N + cj + 0, c20);
    _mm256_store_ps(C + (ci + 2) * N + cj + 8, c21);

    _mm256_store_ps(C + (ci + 3) * N + cj + 0, c30);
    _mm256_store_ps(C + (ci + 3) * N + cj + 8, c31);
}

void matmul(const float *A, const float *B, float *C, float *Bt, int N)
{
    for (int i = 0; i < N; i += 4)
    {
        for (int j = 0; j < N; j += 16)
        {
            matmul_4_16_kernel(A, B, C, N, i, j);
        }
    }
}