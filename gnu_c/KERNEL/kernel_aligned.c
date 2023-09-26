#include "kernel.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

#define RM 6
#define RN 16

typedef float v8sf __attribute__((vector_size(32)));

void matmul(const float *A, const float *B, float *C, int L2M, int L2N, int N)
{
    for (int j = 0; j < N; j += RN)
    {
        for (int i = 0; i < L2M; i += RM)
        {
            v8sf c00 = _mm256_load_ps(C + j * L2M + i * RN + 0);
            v8sf c01 = _mm256_load_ps(C + j * L2M + i * RN + 8);

            v8sf c10 = _mm256_load_ps(C + j * L2M + i * RN + 16);
            v8sf c11 = _mm256_load_ps(C + j * L2M + i * RN + 24);

            v8sf c20 = _mm256_load_ps(C + j * L2M + i * RN + 32);
            v8sf c21 = _mm256_load_ps(C + j * L2M + i * RN + 40);

            v8sf c30 = _mm256_load_ps(C + j * L2M + i * RN + 48);
            v8sf c31 = _mm256_load_ps(C + j * L2M + i * RN + 56);

            v8sf c40 = _mm256_load_ps(C + j * L2M + i * RN + 64);
            v8sf c41 = _mm256_load_ps(C + j * L2M + i * RN + 72);

            v8sf c50 = _mm256_load_ps(C + j * L2M + i * RN + 80);
            v8sf c51 = _mm256_load_ps(C + j * L2M + i * RN + 88);

#pragma GCC unroll 2
            for (int k = 0; k < 128; k++)
            {
                v8sf b0 = _mm256_load_ps(B + j * L2N + k * RN);
                v8sf b1 = _mm256_load_ps(B + j * L2N + k * RN + 8);

                __builtin_prefetch(B + j * L2N + k * RN + 512);

                v8sf a0 = _mm256_broadcast_ss(A + i * L2N + k * RM + 0);
                c00 = _mm256_fmadd_ps(b0, a0, c00);
                c01 = _mm256_fmadd_ps(b1, a0, c01);

                v8sf a1 = _mm256_broadcast_ss(A + i * L2N + k * RM + 1);
                c10 = _mm256_fmadd_ps(b0, a1, c10);
                c11 = _mm256_fmadd_ps(b1, a1, c11);

                v8sf a2 = _mm256_broadcast_ss(A + i * L2N + k * RM + 2);
                c20 = _mm256_fmadd_ps(b0, a2, c20);
                c21 = _mm256_fmadd_ps(b1, a2, c21);

                v8sf a3 = _mm256_broadcast_ss(A + i * L2N + k * RM + 3);
                c30 = _mm256_fmadd_ps(b0, a3, c30);
                c31 = _mm256_fmadd_ps(b1, a3, c31);

                v8sf a4 = _mm256_broadcast_ss(A + i * L2N + k * RM + 4);
                c40 = _mm256_fmadd_ps(b0, a4, c40);
                c41 = _mm256_fmadd_ps(b1, a4, c41);

                v8sf a5 = _mm256_broadcast_ss(A + i * L2N + k * RM + 5);
                c50 = _mm256_fmadd_ps(b0, a5, c50);
                c51 = _mm256_fmadd_ps(b1, a5, c51);
            }

            _mm256_store_ps(C + j * L2M + i * RN + 0, c00);
            _mm256_store_ps(C + j * L2M + i * RN + 8, c01);

            _mm256_store_ps(C + j * L2M + i * RN + 16, c10);
            _mm256_store_ps(C + j * L2M + i * RN + 24, c11);

            _mm256_store_ps(C + j * L2M + i * RN + 32, c20);
            _mm256_store_ps(C + j * L2M + i * RN + 40, c21);

            _mm256_store_ps(C + j * L2M + i * RN + 48, c30);
            _mm256_store_ps(C + j * L2M + i * RN + 56, c31);

            _mm256_store_ps(C + j * L2M + i * RN + 64, c40);
            _mm256_store_ps(C + j * L2M + i * RN + 72, c41);

            _mm256_store_ps(C + j * L2M + i * RN + 80, c50);
            _mm256_store_ps(C + j * L2M + i * RN + 88, c51);
        }
    }
}