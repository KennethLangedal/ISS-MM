#include "MM.h"

#include <stdlib.h>
#include <stdalign.h>
#include <immintrin.h>

#define RM 4
#define RN 16

typedef float v8sf __attribute__((vector_size(32)));

static inline void kernel(const float *A, const float *B, float *C, int N, int K)
{
    v8sf c00 = _mm256_loadu_ps(C);
    v8sf c01 = _mm256_loadu_ps(C + 8);
    v8sf c10 = _mm256_loadu_ps(C + N);
    v8sf c11 = _mm256_loadu_ps(C + N + 8);
    v8sf c20 = _mm256_loadu_ps(C + 2 * N);
    v8sf c21 = _mm256_loadu_ps(C + 2 * N + 8);
    v8sf c30 = _mm256_loadu_ps(C + 3 * N);
    v8sf c31 = _mm256_loadu_ps(C + 3 * N + 8);

    for (int i = 0; i < K; i++)
    {
        v8sf b0 = _mm256_loadu_ps(B + i * N);
        v8sf b1 = _mm256_loadu_ps(B + i * N + 8);

        v8sf a0 = _mm256_broadcast_ss(A + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + K + i);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + 2 * K + i);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + 3 * K + i);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);
    }

    _mm256_storeu_ps(C, c00);
    _mm256_storeu_ps(C + 8, c01);
    _mm256_storeu_ps(C + N, c10);
    _mm256_storeu_ps(C + N + 8, c11);
    _mm256_storeu_ps(C + 2 * N, c20);
    _mm256_storeu_ps(C + 2 * N + 8, c21);
    _mm256_storeu_ps(C + 3 * N, c30);
    _mm256_storeu_ps(C + 3 * N + 8, c31);
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    for (int i = 0; i < M; i += RM)
    {
        for (int j = 0; j < N; j += RN)
        {
            kernel(A + i * K, B + j, C + i * N + j, N, K);
        }
    }
}