/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

#define N SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp) {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

void gesummv_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                 DATA_TYPE *tmp) {
  #pragma omp target map(to : A[ : N *N], B[ : N *N], x[ : N], tmp[ : N]) map(tofrom : y[ : N]) device(OMP_DEVICE_ID)
  #pragma omp teams distribute parallel for
  for (int i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (int j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

void init(DATA_TYPE *A, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

int compareResults(DATA_TYPE *y, DATA_TYPE *y_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < (N); i++) {
    if (percentDiff(y[i], y_outputFromGpu[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  return fail;
}

int main(int argc, char *argv[]) {
  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");

  // declare arrays and allocate memory for common arrays
  DATA_TYPE *A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  DATA_TYPE *x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  DATA_TYPE *tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  DATA_TYPE *y = NULL;
  DATA_TYPE *y_OMP = NULL;
  
  // init common arrays
  init(A, x);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  y_OMP = (DATA_TYPE *)calloc(N, sizeof(DATA_TYPE));
  BENCHMARK_OMP(gesummv_OMP(A, B, x, y_OMP, tmp));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  BENCHMARK_CPU(gesummv(A, B, x, y, tmp));
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail = compareResults(y, y_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // release memory
  free(A);
  free(B);
  free(x);
  free(tmp);
  free(y);
  free(y_OMP);

  return fail;
}
