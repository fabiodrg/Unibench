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

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#ifndef SIZE
#define SIZE 1024
#endif

#define N SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
                DATA_TYPE *y2, DATA_TYPE *x1_gpu, DATA_TYPE *x2_gpu) {
  int i, j;

  for (i = 0; i < N; i++) {
    x1[i] = ((DATA_TYPE)i) / N;
    x2[i] = ((DATA_TYPE)i + 1) / N;
    x1_gpu[i] = x1[i];
    x2_gpu[i] = x2[i];
    y1[i] = ((DATA_TYPE)i + 3) / N;
    y2[i] = ((DATA_TYPE)i + 4) / N;
    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
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
  #pragma omp target teams map(to: a[:N*N], y1[:N], y2[:N]) map(tofrom: x1[:N], x2[:N]) device(DEVICE_ID)                                                    
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
        PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }

    if (percentDiff(x2[i], x2_outputFromGpu[i]) >
        PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);

  return fail;
}

int main() {
  double t_start, t_end;
  int fail = 0;

  /** Variable declaration */
  DATA_TYPE *a;
  DATA_TYPE *x1;
  DATA_TYPE *x2;
  DATA_TYPE *x1_outputFromGpu;
  DATA_TYPE *x2_outputFromGpu;
  DATA_TYPE *y_1;
  DATA_TYPE *y_2;

  /** Memory allocation */
  a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  
  fprintf(stdout, "<< Matrix Vector Product and Transpose >>\n");

  /** Initialize */
  init_array(a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu);

/** Enable all modes if correctness test mode enabled */
#ifdef RUN_TEST
  #define OMP_GPU
  #define CPU_SEQ
  #define N_RUNS 1
#endif

/** Run parallel on GPU */
#ifdef OMP_GPU
  BENCHMARK_GPU(runMvt_OMP(a, x1_outputFromGpu, x2_outputFromGpu, y_1, y_2));
#endif

/** Run sequential on CPU */
#ifdef CPU_SEQ
  BENCHMARK_CPU(runMvt(a, x1, x2, y_1, y_2));
#endif

/** Compare results if correctness test is enabled */
#ifdef RUN_TEST
  fail = compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);
#endif

  /** Release memory */
  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

  return fail;
}
