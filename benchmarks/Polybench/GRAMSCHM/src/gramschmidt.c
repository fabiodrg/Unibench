/**
 * gramschmidt.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <math.h>
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

/* Problem size */
#define M SIZE
#define N SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gramschmidt(DATA_TYPE *A, DATA_TYPE *R, DATA_TYPE *Q) {
  int i, j, k;
  DATA_TYPE nrm;
  for (k = 0; k < N; k++) {
    nrm = 0;
    for (i = 0; i < M; i++) {
      nrm += A[i * N + k] * A[i * N + k];
    }

    R[k * N + k] = sqrt(nrm);
    for (i = 0; i < M; i++) {
      Q[i * N + k] = A[i * N + k] / R[k * N + k];
    }

    for (j = k + 1; j < N; j++) {
      R[k * N + j] = 0;
      for (i = 0; i < M; i++) {
        R[k * N + j] += Q[i * N + k] * A[i * N + j];
      }
      for (i = 0; i < M; i++) {
        A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
      }
    }
  }
}

void gramschmidt_OMP(DATA_TYPE *A, DATA_TYPE *R, DATA_TYPE *Q) {
  int i, j, k;
  DATA_TYPE nrm;

  #pragma omp target data map(to: R[:M*N], Q[:M*N]) map(tofrom: A[:M*N]) device(OMP_DEVICE_ID)
  {
    for (k = 0; k < N; k++) {
      // CPU
      nrm = 0;
      #pragma omp target update from(A[:M*N])
      for (i = 0; i < M; i++) {
        nrm += A[i * N + k] * A[i * N + k];
      }
      R[k * N + k] = sqrt(nrm);
      
      for (i = 0; i < M; i++) {
        Q[i * N + k] = A[i * N + k] / R[k * N + k];
      }
      #pragma omp target update to(Q[:M*N])
      #pragma omp target teams distribute parallel for private(i)
      for (j = k + 1; j < N; j++) {
        R[k * N + j] = 0;
        for (i = 0; i < M; i++) {
          R[k * N + j] += Q[i * N + k] * A[i * N + j];
        }
        for (i = 0; i < M; i++) {
          A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
        }
      }
    }
  }
}

void init_array(DATA_TYPE *A) {
  int i, j;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
    }
  }
}

int compareResults(DATA_TYPE *A, DATA_TYPE *A_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(A[i * N + j], A_outputFromGpu[i * N + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  return fail;
}

int main(int argc, char *argv[]) {
  fprintf(stdout, "<< Gram-Schmidt decomposition >>\n");

  // declare arrays and allocate memory
  DATA_TYPE *A = NULL;
  DATA_TYPE *A_OMP = NULL;
  DATA_TYPE *R = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));
  DATA_TYPE *Q = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));


// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  A_OMP = (DATA_TYPE *) malloc(M * N * sizeof(DATA_TYPE));
  init_array(A_OMP);
  BENCHMARK_OMP(gramschmidt_OMP(A_OMP, R, Q));
  // prevent dead-code elimination
  DCE_PREVENT(A_OMP, M*N);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  A = (DATA_TYPE *) malloc(M * N * sizeof(DATA_TYPE));
  init_array(A);
  BENCHMARK_CPU(gramschmidt(A, R, Q));
  // prevent dead-code elimination
  DCE_PREVENT(A, M*N);
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail = compareResults(A, A_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // release memory
  free(A);
  free(A_OMP);
  free(R);
  free(Q);

  return fail;
}
