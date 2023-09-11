#include "MM.h"
#include <cblas.h>

void matmul(float *A, float *B, float *C, int N)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}