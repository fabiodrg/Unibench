#define _OPS_O2
/**
 * mvt.c: This file was adapted from PolyBench/GPU 1.0 test suite
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

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_common_arrays(DATA_TYPE *A, DATA_TYPE *y1, DATA_TYPE *y2) {
  int i, j;

  for (i = 0; i < N; i++) {
    y1[i] = ((DATA_TYPE)i + 3) / N;
    y2[i] = ((DATA_TYPE)i + 4) / N;
    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void init_vector_x(DATA_TYPE *x1, DATA_TYPE *x2) {
  int i, j;

  for (i = 0; i < N; i++) {
    x1[i] = ((DATA_TYPE)i) / N;
    x2[i] = ((DATA_TYPE)i + 1) / N;
  }
}

void runMvt(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
            DATA_TYPE *y2) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      x1[i] = x1[i] + a[i * N + j] * y1[j];
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      x2[i] = x2[i] + a[j * N + i] * y2[j];
    }
  }
}

void runMvt_OMP(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
                DATA_TYPE *y2) {
  int i, j;
  #pragma omp target teams map(to: a[:N*N], y1[:N], y2[:N]) map(tofrom: x1[:N], x2[:N]) device(OMP_DEVICE_ID)                                                    
  {
    #pragma omp distribute parallel for private(j)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        x1[i] = x1[i] + a[i * N + j] * y1[j];
      }
    }

    #pragma omp distribute parallel for private(j)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        x2[i] = x2[i] + a[j * N + i] * y2[j];
      }
    }
  }
}

int compareResults(DATA_TYPE *x1, DATA_TYPE *x1_outputFromGpu, DATA_TYPE *x2,
                   DATA_TYPE *x2_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < N; i++) {
    if (percentDiff(x1[i], x1_outputFromGpu[i]) >
        ERROR_THRESHOLD) {
      fail++;
    }

    if (percentDiff(x2[i], x2_outputFromGpu[i]) >
        ERROR_THRESHOLD) {
      fail++;
    }
  }

  return fail;
}

int main() {
  SALUTE("Matrix Vector Product and Transpose");

  // Declare arrays and allocate memory for common arrays
  DATA_TYPE *a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  DATA_TYPE *x1 = NULL;
  DATA_TYPE *x2 = NULL;
  DATA_TYPE *x1_OMP = NULL;
  DATA_TYPE *x2_OMP = NULL;
  DATA_TYPE *y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  DATA_TYPE *y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  // Initialize common memory
  init_common_arrays(a, y_1, y_2);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  x1_OMP = (DATA_TYPE *) malloc(N * sizeof(DATA_TYPE));
  x2_OMP = (DATA_TYPE *) malloc(N * sizeof(DATA_TYPE));
  init_vector_x(x1_OMP, x2_OMP);
  BENCHMARK_OMP(runMvt_OMP(a, x1_OMP, x2_OMP, y_1, y_2));
  // prevent dead-code elimination
  DCE_PREVENT(x1_OMP, N);
  DCE_PREVENT(x2_OMP, N);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  x1 = (DATA_TYPE *) malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *) malloc(N * sizeof(DATA_TYPE));
  init_vector_x(x1, x2);
  BENCHMARK_CPU(runMvt(a, x1, x2, y_1, y_2));
  // prevent dead-code elimination
  DCE_PREVENT(x1, N);
  DCE_PREVENT(x2, N);
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail = compareResults(x1, x1_OMP, x2, x2_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // Release memory
  free(a);
  free(x1);
  free(x2);
  free(x1_OMP);
  free(x2_OMP);
  free(y_1);
  free(y_2);

  return fail;
}
