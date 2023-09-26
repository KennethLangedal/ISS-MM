#include "kernel.h"
#include <cblas.h>

void matmul(const float *A, const float *B, float *C, int L2M, int L2N, int N)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                L2M, N, L2N, 1.0f, A, L2N, B, N, 0.0f, C, N);
}