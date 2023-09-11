#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

int main()
{
    srand(0);

    int N = 1 << 26;

    float A = 1.0f;
    float *B = (float *)aligned_alloc(32, N * sizeof(float));
    float *C = (float *)aligned_alloc(32, N * sizeof(float));

    for (int i = 0; i < N; i++)
        B[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;

    for (int i = 0; i < N; i++)
        C[i] = 0.0f;

    for (int n = 64; n < N; n <<= 1)
    {
        int it = N / n;

        clock_t start = clock();

        for (int t = 0; t < it; t++)
        {
            // Scalar
            // for (int i = 0; i < n; i++)
            // {
            //     C[i] = A * B[i];
            // }
            // Vector

            __m256 a = _mm256_broadcast_ss(&A);
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();

            for (int i = 0; i < n; i += 32)
            {
                __m256 b1 = _mm256_load_ps(B + i);
                __m256 b2 = _mm256_load_ps(B + i + 8);
                __m256 b3 = _mm256_load_ps(B + i + 16);
                __m256 b4 = _mm256_load_ps(B + i + 24);

                c1 = _mm256_fmadd_ps(a, b1, c1);
                c2 = _mm256_fmadd_ps(a, b2, c2);
                c3 = _mm256_fmadd_ps(a, b2, c3);
                c4 = _mm256_fmadd_ps(a, b2, c4);
            }

            _mm256_store_ps(C, c1);
            _mm256_store_ps(C + 8, c2);
            _mm256_store_ps(C + 16, c3);
            _mm256_store_ps(C + 24, c4);
        }

        clock_t end = clock();

        double total_duration = (double)(end - start) / CLOCKS_PER_SEC;
        double average_duration = total_duration / it;
        double ops = n * 2;
        double flops = ops / average_duration;

        float cs = 0.0f;
        for (int i = 0; i < n; i++)
            cs += C[i];

        printf("%d %.5f %.5f %.5f\n", n, average_duration, flops / 1e9, cs);
    }

    return 0;
}