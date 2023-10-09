#include "FMA.h"
#include <immintrin.h>

typedef float v8sf __attribute__((vector_size(32)));

const int ops_per_element = 2;

// Manual vectorization
// C[i] = A * B[i] + C[i]

void FMA(const float *a, const float *b, float *c, int n, int it)
{
    for (int t = 0; t < it; t++)
    {
        v8sf va = _mm256_broadcast_ss(a);

        for (int i = 0; i < n; i += 32)
        {
            v8sf vb1 = _mm256_load_ps(b + i);
            v8sf vb2 = _mm256_load_ps(b + i + 8);
            v8sf vb3 = _mm256_load_ps(b + i + 16);
            v8sf vb4 = _mm256_load_ps(b + i + 24);

            v8sf vc1 = _mm256_load_ps(c + i);
            v8sf vc2 = _mm256_load_ps(c + i + 8);
            v8sf vc3 = _mm256_load_ps(c + i + 16);
            v8sf vc4 = _mm256_load_ps(c + i + 24);

            vc1 = _mm256_fmadd_ps(va, vb1, vc1);
            vc2 = _mm256_fmadd_ps(va, vb2, vc2);
            vc3 = _mm256_fmadd_ps(va, vb3, vc3);
            vc4 = _mm256_fmadd_ps(va, vb4, vc4);

            _mm256_store_ps(c + i, vc1);
            _mm256_store_ps(c + i + 8, vc2);
            _mm256_store_ps(c + i + 16, vc3);
            _mm256_store_ps(c + i + 24, vc4);
        }
    }
}