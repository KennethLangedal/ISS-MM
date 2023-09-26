#include "MM.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

// ---- Do not change constants ----

#define RN 16
#define RM 6

#define L2M (RM * 22)
#define L2N 256

// ---------------------------------

typedef float v8sf __attribute__((vector_size(32)));

static inline void kernel(const float *A, const float *B, float *C, int N)
{
    v8sf c00 = _mm256_loadu_ps(C);
    v8sf c01 = _mm256_loadu_ps(C + 8);
    v8sf c10 = _mm256_loadu_ps(C + N);
    v8sf c11 = _mm256_loadu_ps(C + N + 8);
    v8sf c20 = _mm256_loadu_ps(C + 2 * N);
    v8sf c21 = _mm256_loadu_ps(C + 2 * N + 8);
    v8sf c30 = _mm256_loadu_ps(C + 3 * N);
    v8sf c31 = _mm256_loadu_ps(C + 3 * N + 8);
    v8sf c40 = _mm256_loadu_ps(C + 4 * N);
    v8sf c41 = _mm256_loadu_ps(C + 4 * N + 8);
    v8sf c50 = _mm256_loadu_ps(C + 5 * N);
    v8sf c51 = _mm256_loadu_ps(C + 5 * N + 8);

#pragma GCC unroll 8
    for (int i = 0; i < L2N; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * RN);
        v8sf b1 = _mm256_load_ps(B + i * RN + 8);

        v8sf a0 = _mm256_broadcast_ss(A + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + L2N + i);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + 2 * L2N + i);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + 3 * L2N + i);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);

        v8sf a4 = _mm256_broadcast_ss(A + 4 * L2N + i);
        c40 = _mm256_fmadd_ps(b0, a4, c40);
        c41 = _mm256_fmadd_ps(b1, a4, c41);

        v8sf a5 = _mm256_broadcast_ss(A + 5 * L2N + i);
        c50 = _mm256_fmadd_ps(b0, a5, c50);
        c51 = _mm256_fmadd_ps(b1, a5, c51);
    }

    _mm256_storeu_ps(C, c00);
    _mm256_storeu_ps(C + 8, c01);
    _mm256_storeu_ps(C + N, c10);
    _mm256_storeu_ps(C + N + 8, c11);
    _mm256_storeu_ps(C + 2 * N, c20);
    _mm256_storeu_ps(C + 2 * N + 8, c21);
    _mm256_storeu_ps(C + 3 * N, c30);
    _mm256_storeu_ps(C + 3 * N + 8, c31);
    _mm256_storeu_ps(C + 4 * N, c40);
    _mm256_storeu_ps(C + 4 * N + 8, c41);
    _mm256_storeu_ps(C + 5 * N, c50);
    _mm256_storeu_ps(C + 5 * N + 8, c51);
}

/*
    Computes C += A * B

    Every matrix should be stored in row-major order.

    The dimentions should be as follows:
    C: M x N
    A: M x K
    B: K x N

    So C should have M rows and N columns

    For best performance, let:
    N = 16 * x
    M = 132 * y
    K = 256 * z
*/
void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    float *L2A = (float *)aligned_alloc(32, sizeof(float) * L2M * L2N);
    float *L1B = (float *)aligned_alloc(32, sizeof(float) * L2N * RN);

    for (int i = 0; i < M; i += L2M)
    {
        for (int k = 0; k < K; k += L2N)
        {
            // Copy A to local aligned L2 space
            for (int ai = 0; ai < L2M; ai++)
                for (int ak = 0; ak < L2N; ak++)
                    L2A[ai * L2N + ak] = A[(i + ai) * K + k + ak];

            for (int j = 0; j < N; j += RN)
            {
                // Copy B to local aligned L1 space
                for (int bk = 0; bk < L2N; bk++)
                    for (int bj = 0; bj < RN; bj++)
                        L1B[bk * RN + bj] = B[(k + bk) * N + j + bj];

                // Call kernel for L2 tile of A, L1 tile of B, and every register sized tile of C
                for (int ci = 0; ci < L2M; ci += RM)
                    kernel(L2A + ci * L2N, L1B, C + (i + ci) * N + j, N);
            }
        }
    }

    free(L2A);
    free(L1B);
}