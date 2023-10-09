#include "FMA.h"
#include <immintrin.h>

typedef float v8sf __attribute__((vector_size(32)));

const int ops_per_element = 2;

// Manual vectorization
// C[i] = A[i] * B[i] + C[i]

void FMA(const float *a, const float *b, float *c, int n, int it)
{
    for (int t = 0; t < it; t++)
    {
        for (int i = 0; i < n; i += 32)
        {
            v8sf va1 = _mm256_load_ps(a + i);
            v8sf va2 = _mm256_load_ps(a + i + 8);
            v8sf va3 = _mm256_load_ps(a + i + 16);
            v8sf va4 = _mm256_load_ps(a + i + 24);

            v8sf vb1 = _mm256_load_ps(b + i);
            v8sf vb2 = _mm256_load_ps(b + i + 8);
            v8sf vb3 = _mm256_load_ps(b + i + 16);
            v8sf vb4 = _mm256_load_ps(b + i + 24);

            v8sf vc1 = _mm256_load_ps(c + i);
            v8sf vc2 = _mm256_load_ps(c + i + 8);
            v8sf vc3 = _mm256_load_ps(c + i + 16);
            v8sf vc4 = _mm256_load_ps(c + i + 24);

            vc1 = _mm256_fmadd_ps(va1, vb1, vc1);
            vc2 = _mm256_fmadd_ps(va2, vb2, vc2);
            vc3 = _mm256_fmadd_ps(va3, vb3, vc3);
            vc4 = _mm256_fmadd_ps(va4, vb4, vc4);

            _mm256_store_ps(c + i, vc1);
            _mm256_store_ps(c + i + 8, vc2);
            _mm256_store_ps(c + i + 16, vc3);
            _mm256_store_ps(c + i + 24, vc4);
        }
    }
}