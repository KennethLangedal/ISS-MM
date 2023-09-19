#include "MM.h"
#include <cblas.h>

void matmul(const float *A, const float *B, float *C, float *Bt, int N)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}