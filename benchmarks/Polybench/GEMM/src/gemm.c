/**
 * gemm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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

#define NI SIZE
#define NJ SIZE
#define NK SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  }
}

void init_C(DATA_TYPE *C) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  }
}

int compareResults(DATA_TYPE *C, DATA_TYPE *C_OMP) {
  int i, j, fail;
  fail = 0;

  // Compare C1 and C2
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      if (percentDiff(C[i * NJ + j], C_OMP[i * NJ + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  return fail;
}

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

void gemm_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  #pragma omp target map(to : A[ : NI *NK], B[ : NK *NJ]) map(tofrom : C[ : NI *NJ]) device(OMP_DEVICE_ID)
  #pragma omp teams distribute parallel for
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;
      for (int k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  SALUTE("Matrix-multiply C=alpha.A.B+beta.C");

  // declare arrays and allocate memory for common arrays
  DATA_TYPE *A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  DATA_TYPE *C = NULL;
  DATA_TYPE *C_OMP = NULL;

  // init common arrays
  init(A, B);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  C_OMP = (DATA_TYPE *) calloc(NI * NJ, sizeof(DATA_TYPE));
  init_C(C_OMP);
  BENCHMARK_OMP(gemm_OMP(A, B, C_OMP));
  // prevent dead-code elimination
  DCE_PREVENT(C_OMP, NI*NJ);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  C = (DATA_TYPE *) calloc(NI * NJ, sizeof(DATA_TYPE));
  init_C(C);
  BENCHMARK_CPU(gemm(A, B, C));
  // prevent dead-code elimination
  DCE_PREVENT(C, NI*NJ);
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail = compareResults(C, C_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif
  
  // release memory
  free(A);
  free(B);
  free(C);
  free(C_OMP);

  return fail;
}
