#include "MM.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

#define RN 16 // Do not change
#define RM 5  // Do not change

#define L2M (RM * 40)
#define L2N 300

typedef float v8sf __attribute__((vector_size(32)));

static inline void kernel(const float *A, const float *B, float *C, int ci, int cj, int N)
{
    v8sf c00 = _mm256_loadu_ps(C + N * (ci + 0) + cj);
    v8sf c01 = _mm256_loadu_ps(C + N * (ci + 0) + cj + 8);
    v8sf c10 = _mm256_loadu_ps(C + N * (ci + 1) + cj);
    v8sf c11 = _mm256_loadu_ps(C + N * (ci + 1) + cj + 8);
    v8sf c20 = _mm256_loadu_ps(C + N * (ci + 2) + cj);
    v8sf c21 = _mm256_loadu_ps(C + N * (ci + 2) + cj + 8);
    v8sf c30 = _mm256_loadu_ps(C + N * (ci + 3) + cj);
    v8sf c31 = _mm256_loadu_ps(C + N * (ci + 3) + cj + 8);
    v8sf c40 = _mm256_loadu_ps(C + N * (ci + 4) + cj);
    v8sf c41 = _mm256_loadu_ps(C + N * (ci + 4) + cj + 8);
    // v8sf c50 = _mm256_loadu_ps(C + N * (ci + 5) + cj);
    // v8sf c51 = _mm256_loadu_ps(C + N * (ci + 5) + cj + 8);

    for (int i = 0; i < L2N; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * RN + 0);
        v8sf b1 = _mm256_load_ps(B + i * RN + 8);

        v8sf a0 = _mm256_broadcast_ss(A + 0 * L2N + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + 1 * L2N + i);
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

        // v8sf a5 = _mm256_broadcast_ss(A + 5 * L2N + i);
        // c50 = _mm256_fmadd_ps(b0, a5, c50);
        // c51 = _mm256_fmadd_ps(b1, a5, c51);
    }

    _mm256_storeu_ps(C + N * (ci + 0) + cj, c00);
    _mm256_storeu_ps(C + N * (ci + 0) + cj + 8, c01);
    _mm256_storeu_ps(C + N * (ci + 1) + cj, c10);
    _mm256_storeu_ps(C + N * (ci + 1) + cj + 8, c11);
    _mm256_storeu_ps(C + N * (ci + 2) + cj, c20);
    _mm256_storeu_ps(C + N * (ci + 2) + cj + 8, c21);
    _mm256_storeu_ps(C + N * (ci + 3) + cj, c30);
    _mm256_storeu_ps(C + N * (ci + 3) + cj + 8, c31);
    _mm256_storeu_ps(C + N * (ci + 4) + cj, c40);
    _mm256_storeu_ps(C + N * (ci + 4) + cj + 8, c41);
    // _mm256_storeu_ps(C + N * (ci + 5) + cj, c50);
    // _mm256_storeu_ps(C + N * (ci + 5) + cj + 8, c51);
}

// M should divide L2M
// N should divide RN
// K should divide L2N

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
            // Copy A to local alligned area
            for (int ai = 0; ai < L2M; ai++)
                for (int ak = 0; ak < L2N; ak++)
                    L2A[ai * L2N + ak] = A[(i + ai) * K + k + ak];

            for (int j = 0; j < N; j += RN)
            {
                // Copy B to local L1 space
                for (int bk = 0; bk < L2N; bk++)
                    for (int bj = 0; bj < RN; bj++)
                        L1B[bk * RN + bj] = B[(k + bk) * N + j + bj];

                // Call kernel for A_L2, B_L1, and every register sized pice of C
                for (int ci = 0; ci < L2M; ci += RM)
                {
                    kernel(L2A + ci * L2N, L1B, C, i + ci, j, N);
                }
            }
        }
    }

    free(L2A);
    free(L1B);
}