#include "kernel.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

#define RM 4
#define RN 16

typedef float v8sf __attribute__((vector_size(32)));

void matmul(const float *A, const float *B, float *C, int L2M, int L2N, int N)
{
    float *L1B = (float *)aligned_alloc(32, sizeof(float) * L2N * RN);

    for (int j = 0; j < N; j += RN)
    {
        // Copy B to local L1 space
        for (int bk = 0; bk < L2N; bk++)
            for (int bj = 0; bj < RN; bj++)
                L1B[bk * RN + bj] = B[bk * N + j + bj];

        for (int i = 0; i < L2M; i += RM)
        {
            v8sf c00 = _mm256_load_ps(C + (0 + i) * N + j + 0);
            v8sf c01 = _mm256_load_ps(C + (0 + i) * N + j + 8);
            // v8sf c02 = _mm256_load_ps(C + (0 + i) * N + j + 16);

            v8sf c10 = _mm256_load_ps(C + (1 + i) * N + j + 0);
            v8sf c11 = _mm256_load_ps(C + (1 + i) * N + j + 8);
            // v8sf c12 = _mm256_load_ps(C + (1 + i) * N + j + 16);

            v8sf c20 = _mm256_load_ps(C + (2 + i) * N + j + 0);
            v8sf c21 = _mm256_load_ps(C + (2 + i) * N + j + 8);
            // v8sf c22 = _mm256_load_ps(C + (2 + i) * N + j + 16);

            v8sf c30 = _mm256_load_ps(C + (3 + i) * N + j + 0);
            v8sf c31 = _mm256_load_ps(C + (3 + i) * N + j + 8);
            // v8sf c32 = _mm256_load_ps(C + (3 + i) * N + j + 16);

            for (int k = 0; k < L2N; k++)
            {
                v8sf b0 = _mm256_load_ps(L1B + k * RN);
                v8sf b1 = _mm256_load_ps(L1B + k * RN + 8);
                // v8sf b2 = _mm256_load_ps(L1B + k * RN + 16);

                v8sf a0 = _mm256_broadcast_ss(A + (i + 0) * L2N + k);
                c00 = _mm256_fmadd_ps(b0, a0, c00);
                c01 = _mm256_fmadd_ps(b1, a0, c01);
                // c02 = _mm256_fmadd_ps(b2, a0, c02);

                v8sf a1 = _mm256_broadcast_ss(A + (i + 1) * L2N + k);
                c10 = _mm256_fmadd_ps(b0, a1, c10);
                c11 = _mm256_fmadd_ps(b1, a1, c11);
                // c12 = _mm256_fmadd_ps(b2, a1, c12);

                v8sf a2 = _mm256_broadcast_ss(A + (i + 2) * L2N + k);
                c20 = _mm256_fmadd_ps(b0, a2, c20);
                c21 = _mm256_fmadd_ps(b1, a2, c21);
                // c22 = _mm256_fmadd_ps(b2, a2, c22);

                v8sf a3 = _mm256_broadcast_ss(A + (i + 3) * L2N + k);
                c30 = _mm256_fmadd_ps(b0, a3, c30);
                c31 = _mm256_fmadd_ps(b1, a3, c31);
                // c32 = _mm256_fmadd_ps(b2, a3, c32);
            }

            _mm256_store_ps(C + (0 + i) * N + j + 0, c00);
            _mm256_store_ps(C + (0 + i) * N + j + 8, c01);
            // _mm256_store_ps(C + (0 + i) * N + j + 16, c02);

            _mm256_store_ps(C + (1 + i) * N + j + 0, c10);
            _mm256_store_ps(C + (1 + i) * N + j + 8, c11);
            // _mm256_store_ps(C + (1 + i) * N + j + 16, c12);

            _mm256_store_ps(C + (2 + i) * N + j + 0, c20);
            _mm256_store_ps(C + (2 + i) * N + j + 8, c21);
            // _mm256_store_ps(C + (2 + i) * N + j + 16, c22);

            _mm256_store_ps(C + (3 + i) * N + j + 0, c30);
            _mm256_store_ps(C + (3 + i) * N + j + 8, c31);
            // _mm256_store_ps(C + (3 + i) * N + j + 16, c32);
        }
    }

    free(L1B);
}