#include "MM.h"

#include <immintrin.h>

typedef float v8sf __attribute__((vector_size(32)));

static inline void matmul_4_16_4_kernel(const float *A, const float *B, float *C,
                                        int N, int ci, int cj, int ck)
{
    for (int i = 0; i < 4; i++)
    {
        for (int k = 0; k < 4; k++)
        {
            for (int j = 0; j < 16; j++)
            {
                C[(ci + i) * N + cj + j] += A[(ci + i) * N + ck + k] * B[(ck + k) * N + cj + j];
            }
        }
    }
}

static inline void matmul_4_16_kernel_vec(const float *A, const float *B, float *C,
                                          int N, int ci, int cj, int l, int r)
{
    v8sf c00 = _mm256_load_ps(C + (ci + 0) * N + cj + 0);
    v8sf c01 = _mm256_load_ps(C + (ci + 0) * N + cj + 8);

    v8sf c10 = _mm256_load_ps(C + (ci + 1) * N + cj + 0);
    v8sf c11 = _mm256_load_ps(C + (ci + 1) * N + cj + 8);

    v8sf c20 = _mm256_load_ps(C + (ci + 2) * N + cj + 0);
    v8sf c21 = _mm256_load_ps(C + (ci + 2) * N + cj + 8);

    v8sf c30 = _mm256_load_ps(C + (ci + 3) * N + cj + 0);
    v8sf c31 = _mm256_load_ps(C + (ci + 3) * N + cj + 8);

    for (int k = l; k < r; k++)
    {
        const float *_A = A + ci * N + k;

        v8sf b0 = _mm256_load_ps(B + k * N + cj);
        v8sf b1 = _mm256_load_ps(B + k * N + cj + 8);

        v8sf a0 = _mm256_broadcast_ss(_A + N * 0);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(_A + N * 1);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(_A + N * 2);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(_A + N * 3);
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

#define COLS 64
#define ROWS 32
#define BLOCKS 32

void matmul(const float *A, const float *B, float *C, float *Bt, int N)
{
    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;

    // Fix A
    int p = 0;
    for (int I = 0; I < N; I += ROWS)
        for (int i = 0; i < ROWS; i++)
            for (int k = 0; k < BLOCKS; k++)
                for (int t = 0; t < 4; t++)
                    Bt[p++] = A[(I + i + t) * N + k];

    for (int I = 0; I < N; I += ROWS)
    {
        for (int J = 0; J < N; J += COLS)
        {
            for (int K = 0; K < N; K += BLOCKS)
            {
                for (int i = 0; i < ROWS; i += 4)
                {
                    for (int j = 0; j < COLS; j += 16)
                    {
                        matmul_4_16_kernel_vec(A, B, C, N, I + i, J + j, K, K + BLOCKS);
                    }
                }
            }
        }
    }
}