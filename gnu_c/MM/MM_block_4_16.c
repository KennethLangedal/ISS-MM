#include "MM.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

#define N_L2 256

#define N_L1 16 // Do not change
#define M_R 4   // Do not change

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

    for (int i = 0; i < N_L2; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * N_L1 + 0);
        v8sf b1 = _mm256_load_ps(B + i * N_L1 + 8);

        v8sf a0 = _mm256_broadcast_ss(A + 0 * N_L2 + i);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + 1 * N_L2 + i);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + 2 * N_L2 + i);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + 3 * N_L2 + i);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);
    }

    _mm256_storeu_ps(C + N * (ci + 0) + cj, c00);
    _mm256_storeu_ps(C + N * (ci + 0) + cj + 8, c01);
    _mm256_storeu_ps(C + N * (ci + 1) + cj, c10);
    _mm256_storeu_ps(C + N * (ci + 1) + cj + 8, c11);
    _mm256_storeu_ps(C + N * (ci + 2) + cj, c20);
    _mm256_storeu_ps(C + N * (ci + 2) + cj + 8, c21);
    _mm256_storeu_ps(C + N * (ci + 3) + cj, c30);
    _mm256_storeu_ps(C + N * (ci + 3) + cj + 8, c31);
}

void matmul(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0f;

    float *A_L2 = (float *)aligned_alloc(32, sizeof(float) * N_L2 * N_L2);
    float *B_L1 = (float *)aligned_alloc(32, sizeof(float) * N_L2 * N_L1);

    for (int i = 0; i < N; i += N_L2)
    {
        for (int k = 0; k < N; k += N_L2)
        {
            // Copy A to local alligned area
            for (int ai = 0; ai < N_L2; ai++)
                for (int ak = 0; ak < N_L2; ak++)
                    A_L2[ai * N_L2 + ak] = A[(i + ai) * N + k + ak];

            for (int j = 0; j < N; j += N_L1)
            {
                // Copy B to local L1 space
                for (int bk = 0; bk < N_L2; bk++)
                    for (int bj = 0; bj < N_L1; bj++)
                        B_L1[bk * N_L1 + bj] = B[(k + bk) * N + j + bj];

                // Call kernel for A_L2, B_L1, and every register sized pice of C
                for (int ci = 0; ci < N_L2; ci += M_R)
                {
                    kernel(A_L2 + ci * N_L2, B_L1, C, i + ci, j, N);
                }
            }
        }
    }

    free(A_L2);
    free(B_L1);
}