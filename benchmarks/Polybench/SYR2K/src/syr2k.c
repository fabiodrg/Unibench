/**
 * syr2k.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

#define N SIZE
#define M SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
      B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
    }
  }
}

void init_C_array(DATA_TYPE *C) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
    }
  }
}

void syr2k(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
        C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

void syr2k_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
    }
  }

  #pragma omp target teams map(to : A[ : N *M], B[ : N *M]) map(tofrom : C[ : N *N]) device(OMP_DEVICE_ID)
  #pragma omp distribute parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
        C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

int compareResults(DATA_TYPE *C, DATA_TYPE *C_OMP) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(C[i * N + j], C_OMP[i * N + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  return fail;
}

int main() {
  SALUTE("Symmetric rank-2k operations");
  
  // Declare arrays and allocate memory for common arrays
  DATA_TYPE *A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  DATA_TYPE *C = NULL;
  DATA_TYPE *C_OMP = NULL;

  // Initialize common arrays
  init_arrays(A, B);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  C_OMP = (DATA_TYPE *) calloc(N * M, sizeof(DATA_TYPE));
  init_C_array(C_OMP);
  BENCHMARK_OMP(syr2k_OMP(A, B, C_OMP));
  // prevent dead-code elimination
  DCE_PREVENT(C_OMP, N*M);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  C = (DATA_TYPE *) calloc(N * M, sizeof(DATA_TYPE));
  init_C_array(C);
  BENCHMARK_CPU(syr2k(A, B, C));
  // prevent dead-code elimination
  DCE_PREVENT(C, N*M);
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail = compareResults(C, C_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  free(A);
  free(B);
  free(C);
  free(C_OMP);

  return fail;
}
