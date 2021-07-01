/**
 * syrk.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include "BenchmarksUtil.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

/* Problem size */
#define N SIZE
#define M SIZE

/* Declared constant values for alpha and beta */
/* (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array_A(DATA_TYPE *A) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A[i * M + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void init_array_C(DATA_TYPE *C) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
    }
  }
}

int compareResults(DATA_TYPE *C, DATA_TYPE *C_OMP) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      if (percentDiff(C[i * M + j], C_OMP[i * M + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  return fail;
}

void syrk(DATA_TYPE *A, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      C[i * M + j] *= beta;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
      }
    }
  }
}

void syrkOMP(DATA_TYPE *A, DATA_TYPE *C) {
  #pragma omp target teams map(to : A[:N*M]) map(tofrom : C[:N*M]) device(OMP_DEVICE_ID)
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        C[i * M + j] *= beta;
      }
    }
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        for (int k = 0; k < M; k++) {
          C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
        }
      }
    }
  }
}

int main() {
  fprintf(stdout, "<< Symmetric rank-k operations >>\n");

  // declare arrays and allocate memory for common arrays
  DATA_TYPE *A = (DATA_TYPE *) malloc(N * M * sizeof(DATA_TYPE));
  DATA_TYPE *C = NULL;
  DATA_TYPE *C_OMP = NULL;

  // init array A
  init_array_A(A);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  C_OMP = (DATA_TYPE *) malloc(N * M * sizeof(DATA_TYPE));
  init_array_C(C_OMP);
  BENCHMARK_OMP(syrkOMP(A, C_OMP));
   // prevent dead-code elimination
  DCE_PREVENT(C_OMP, N*M);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  C = (DATA_TYPE *) malloc(N * M * sizeof(DATA_TYPE));
  init_array_C(C);
  BENCHMARK_CPU(syrk(A, C));
  // prevent dead-code elimination
  DCE_PREVENT(C, N*M);
#endif

  int fail = 0;
// if TEST is enabled, then compare OMP results against sequential mode
#ifdef RUN_TEST
  fail = compareResults(C, C_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // release memory
  free(A);
  free(C);
  free(C_OMP);

  return fail;
}
