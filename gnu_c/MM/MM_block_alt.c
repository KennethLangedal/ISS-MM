#include "MM.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

#define RN 16 // Do not change
#define RM 6  // Do not change

#define L2M (RM * 36)
#define L2N 200

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

    for (int i = 0; i < L2N; i++)
    {
        v8sf b0 = _mm256_load_ps(B + i * RN);
        v8sf b1 = _mm256_load_ps(B + i * RN + 8);

        v8sf a0 = _mm256_broadcast_ss(A + i * RM);
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        v8sf a1 = _mm256_broadcast_ss(A + i * RM + 1);
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        v8sf a2 = _mm256_broadcast_ss(A + i * RM + 2);
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        v8sf a3 = _mm256_broadcast_ss(A + i * RM + 3);
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);

        v8sf a4 = _mm256_broadcast_ss(A + i * RM + 4);
        c40 = _mm256_fmadd_ps(b0, a4, c40);
        c41 = _mm256_fmadd_ps(b1, a4, c41);

        v8sf a5 = _mm256_broadcast_ss(A + i * RM + 5);
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

// M should divide L2M
// N should divide RN
// K should divide L2N

void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    float *L2A = (float *)aligned_alloc(32, sizeof(float) * L2M * L2N);
    float *L3B = (float *)aligned_alloc(32, sizeof(float) * L2N * N);

    for (int k = 0; k < K; k += L2N)
    {
        // Copy B to local L3 space
        int p = 0;
        for (int bj = 0; bj < N; bj += RN)
            for (int bi = 0; bi < L2N; bi++)
                for (int bk = 0; bk < RN; bk++)
                    L3B[p++] = B[(k + bi) * N + bj + bk];

        for (int i = 0; i < M; i += L2M)
        {
            // Copy A to local alligned area
            p = 0;
            for (int ai = 0; ai < L2M; ai += RM)
                for (int aj = 0; aj < L2N; aj++)
                    for (int ak = 0; ak < RM; ak++)
                        L2A[p++] = A[(i + ai + ak) * K + k + aj];

            for (int j = 0; j < N; j += RN)
            {
                // Call kernel for A, B_L3, and every register sized pice of C
                for (int ci = 0; ci < L2M; ci += RM)
                {
                    kernel(L2A + ci * L2N, L3B + j * L2N, C + (i + ci) * N + j, N);
                }
            }
        }
    }

    free(L2A);
    free(L3B);
}