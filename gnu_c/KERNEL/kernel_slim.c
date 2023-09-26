#include "kernel.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdalign.h>

#define RM 12
#define RN 8

typedef float v8sf __attribute__((vector_size(32)));

void matmul(const float *A, const float *B, float *C, int L2M, int L2N, int N)
{
    for (int j = 0; j < N; j += RN)
    {
        for (int i = 0; i < L2M; i += RM)
        {
            v8sf c0 = _mm256_load_ps(C + j * L2M + i * RN + 0);
            v8sf c1 = _mm256_load_ps(C + j * L2M + i * RN + 8);
            v8sf c2 = _mm256_load_ps(C + j * L2M + i * RN + 16);
            v8sf c3 = _mm256_load_ps(C + j * L2M + i * RN + 24);
            v8sf c4 = _mm256_load_ps(C + j * L2M + i * RN + 32);
            v8sf c5 = _mm256_load_ps(C + j * L2M + i * RN + 40);
            v8sf c6 = _mm256_load_ps(C + j * L2M + i * RN + 48);
            v8sf c7 = _mm256_load_ps(C + j * L2M + i * RN + 56);
            v8sf c8 = _mm256_load_ps(C + j * L2M + i * RN + 64);
            v8sf c9 = _mm256_load_ps(C + j * L2M + i * RN + 72);
            v8sf c10 = _mm256_load_ps(C + j * L2M + i * RN + 80);
            v8sf c11 = _mm256_load_ps(C + j * L2M + i * RN + 88);

            for (int k = 0; k < L2N; k++)
            {
                v8sf b = _mm256_load_ps(B + j * L2N + k * RN);

                v8sf a0 = _mm256_broadcast_ss(A + i * L2N + k * RM + 0);
                c0 = _mm256_fmadd_ps(b, a0, c0);
                v8sf a1 = _mm256_broadcast_ss(A + i * L2N + k * RM + 1);
                c1 = _mm256_fmadd_ps(b, a1, c1);
                v8sf a2 = _mm256_broadcast_ss(A + i * L2N + k * RM + 2);
                c2 = _mm256_fmadd_ps(b, a2, c2);
                v8sf a3 = _mm256_broadcast_ss(A + i * L2N + k * RM + 3);
                c3 = _mm256_fmadd_ps(b, a3, c3);
                v8sf a4 = _mm256_broadcast_ss(A + i * L2N + k * RM + 4);
                c4 = _mm256_fmadd_ps(b, a4, c4);
                v8sf a5 = _mm256_broadcast_ss(A + i * L2N + k * RM + 5);
                c5 = _mm256_fmadd_ps(b, a5, c5);
                v8sf a6 = _mm256_broadcast_ss(A + i * L2N + k * RM + 6);
                c6 = _mm256_fmadd_ps(b, a6, c6);
                v8sf a7 = _mm256_broadcast_ss(A + i * L2N + k * RM + 7);
                c7 = _mm256_fmadd_ps(b, a7, c7);
                v8sf a8 = _mm256_broadcast_ss(A + i * L2N + k * RM + 8);
                c8 = _mm256_fmadd_ps(b, a8, c8);
                v8sf a9 = _mm256_broadcast_ss(A + i * L2N + k * RM + 9);
                c9 = _mm256_fmadd_ps(b, a9, c9);
                v8sf a10 = _mm256_broadcast_ss(A + i * L2N + k * RM + 10);
                c10 = _mm256_fmadd_ps(b, a10, c10);
                v8sf a11 = _mm256_broadcast_ss(A + i * L2N + k * RM + 11);
                c11 = _mm256_fmadd_ps(b, a11, c11);
            }

            _mm256_store_ps(C + j * L2M + i * RN + 0, c0);
            _mm256_store_ps(C + j * L2M + i * RN + 8, c1);
            _mm256_store_ps(C + j * L2M + i * RN + 16, c2);
            _mm256_store_ps(C + j * L2M + i * RN + 24, c3);
            _mm256_store_ps(C + j * L2M + i * RN + 32, c4);
            _mm256_store_ps(C + j * L2M + i * RN + 40, c5);
            _mm256_store_ps(C + j * L2M + i * RN + 48, c6);
            _mm256_store_ps(C + j * L2M + i * RN + 56, c7);
            _mm256_store_ps(C + j * L2M + i * RN + 64, c8);
            _mm256_store_ps(C + j * L2M + i * RN + 72, c9);
            _mm256_store_ps(C + j * L2M + i * RN + 80, c10);
            _mm256_store_ps(C + j * L2M + i * RN + 88, c11);
        }
    }
}