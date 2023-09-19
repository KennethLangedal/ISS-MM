#include "FMA.h"
#include <immintrin.h>

#define UNROLL

typedef float v8sf __attribute__((vector_size(32)));

void FMA(const float *a, const float *b, float *c, int n, int it)
{
    v8sf va = _mm256_broadcast_ss(a);

    v8sf vc1 = _mm256_setzero_ps();
    v8sf vc2 = _mm256_setzero_ps();
    v8sf vc3 = _mm256_setzero_ps();
    v8sf vc4 = _mm256_setzero_ps();
    v8sf vc5 = _mm256_setzero_ps();
    v8sf vc6 = _mm256_setzero_ps();
    v8sf vc7 = _mm256_setzero_ps();
    v8sf vc8 = _mm256_setzero_ps();

    for (int t = 0; t < it; t++)
    {

        for (int i = 0; i < n; i += 64)
        {
            v8sf vb1 = _mm256_load_ps(b + i);
            v8sf vb2 = _mm256_load_ps(b + i + 8);
            v8sf vb3 = _mm256_load_ps(b + i + 16);
            v8sf vb4 = _mm256_load_ps(b + i + 24);
            v8sf vb5 = _mm256_load_ps(b + i + 32);
            v8sf vb6 = _mm256_load_ps(b + i + 40);
            v8sf vb7 = _mm256_load_ps(b + i + 48);
            v8sf vb8 = _mm256_load_ps(b + i + 56);

            vc1 = _mm256_fmadd_ps(va, vb1, vc1);
            vc2 = _mm256_fmadd_ps(va, vb2, vc2);
            vc3 = _mm256_fmadd_ps(va, vb3, vc3);
            vc4 = _mm256_fmadd_ps(va, vb4, vc4);
            vc5 = _mm256_fmadd_ps(va, vb5, vc5);
            vc6 = _mm256_fmadd_ps(va, vb6, vc6);
            vc7 = _mm256_fmadd_ps(va, vb7, vc7);
            vc8 = _mm256_fmadd_ps(va, vb8, vc8);
        }
    }

    _mm256_store_ps(c, vc1);
    _mm256_store_ps(c + 8, vc2);
    _mm256_store_ps(c + 16, vc3);
    _mm256_store_ps(c + 24, vc4);
    _mm256_store_ps(c + 32, vc5);
    _mm256_store_ps(c + 40, vc6);
    _mm256_store_ps(c + 48, vc7);
    _mm256_store_ps(c + 56, vc8);
}