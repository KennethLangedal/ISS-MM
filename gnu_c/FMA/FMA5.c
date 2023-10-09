#include "FMA.h"
#include <immintrin.h>

typedef float v8sf __attribute__((vector_size(32)));

const int ops_per_element = 4;

// Manual vectorization
// C1 = A1 * B[i] + C1
// C2 = A2 * B[i] + C2

void FMA(const float *a, const float *b, float *c, int n, int it)
{
    v8sf va1 = _mm256_broadcast_ss(a);
    v8sf va2 = _mm256_broadcast_ss(a + 1);

    v8sf vc11 = _mm256_setzero_ps();
    v8sf vc12 = _mm256_setzero_ps();
    v8sf vc21 = _mm256_setzero_ps();
    v8sf vc22 = _mm256_setzero_ps();
    v8sf vc31 = _mm256_setzero_ps();
    v8sf vc32 = _mm256_setzero_ps();
    v8sf vc41 = _mm256_setzero_ps();
    v8sf vc42 = _mm256_setzero_ps();

    for (int t = 0; t < it; t++)
    {
        for (int i = 0; i < n; i += 64)
        {
            v8sf vb1 = _mm256_load_ps(b + i);
            v8sf vb2 = _mm256_load_ps(b + i + 8);
            v8sf vb3 = _mm256_load_ps(b + i + 16);
            v8sf vb4 = _mm256_load_ps(b + i + 24);

            vc11 = _mm256_fmadd_ps(va1, vb1, vc11);
            vc12 = _mm256_fmadd_ps(va2, vb1, vc12);
            vc21 = _mm256_fmadd_ps(va1, vb2, vc21);
            vc22 = _mm256_fmadd_ps(va2, vb2, vc22);
            vc31 = _mm256_fmadd_ps(va1, vb3, vc31);
            vc32 = _mm256_fmadd_ps(va2, vb3, vc32);
            vc41 = _mm256_fmadd_ps(va1, vb4, vc41);
            vc42 = _mm256_fmadd_ps(va2, vb4, vc42);

            vb1 = _mm256_load_ps(b + i + 32);
            vb2 = _mm256_load_ps(b + i + 40);
            vb3 = _mm256_load_ps(b + i + 48);
            vb4 = _mm256_load_ps(b + i + 56);

            vc11 = _mm256_fmadd_ps(va1, vb1, vc11);
            vc12 = _mm256_fmadd_ps(va2, vb1, vc12);
            vc21 = _mm256_fmadd_ps(va1, vb2, vc21);
            vc22 = _mm256_fmadd_ps(va2, vb2, vc22);
            vc31 = _mm256_fmadd_ps(va1, vb3, vc31);
            vc32 = _mm256_fmadd_ps(va2, vb3, vc32);
            vc41 = _mm256_fmadd_ps(va1, vb4, vc41);
            vc42 = _mm256_fmadd_ps(va2, vb4, vc42);
        }
    }

    _mm256_store_ps(c, vc11);
    _mm256_store_ps(c + 8, vc12);
    _mm256_store_ps(c + 16, vc21);
    _mm256_store_ps(c + 24, vc22);
    _mm256_store_ps(c + 32, vc31);
    _mm256_store_ps(c + 40, vc32);
    _mm256_store_ps(c + 48, vc41);
    _mm256_store_ps(c + 56, vc42);
}