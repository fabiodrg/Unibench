/**
 * 2DConvolution.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#ifndef SIZE
#define SIZE 1024
#endif

#define NI SIZE
#define NJ SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv2D(DATA_TYPE *A, DATA_TYPE *B) {
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  for (int i = 1; i < NI - 1; ++i) // 0
  {
    for (int j = 1; j < NJ - 1; ++j) // 1
    {
      B[i * NJ + j] =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
          c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
          c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] +
          c33 * A[(i + 1) * NJ + (j + 1)];
    }
  }
}

void conv2D_OMP(DATA_TYPE *A, DATA_TYPE *B) {
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  #pragma omp target teams distribute parallel for map(to : A[ : NI *NJ]) map(from : B[ : NI *NJ]) device(OMP_DEVICE_ID)
  for (int i = 1; i < NI - 1; ++i) {
    LLVM_MCA_BEGIN("loop_j");
    for (int j = 1; j < NJ - 1; ++j) {
      B[i * NJ + j] =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
          c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
          c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] +
          c33 * A[(i + 1) * NJ + (j + 1)];
    }
    LLVM_MCA_END("loop_j");
  }
}

void init(DATA_TYPE *A) {
  int i, j;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      A[i * NJ + j] = (float)rand() / RAND_MAX;
    }
  }
}

int compareResults(DATA_TYPE *B, DATA_TYPE *B_OMP) {
  int i, j, fail;
  fail = 0;

  // Compare B and B_OMP
  for (i = 1; i < (NI - 1); i++) {
    for (j = 1; j < (NJ - 1); j++) {
      if (percentDiff(B[i * NJ + j], B_OMP[i * NJ + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  return fail;
}

int main(int argc, char *argv[]) {
  double t_start, t_end, t_start_OMP, t_end_OMP;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_OMP;

  A = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B_OMP = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two dimensional (2D) convolution <<\n");

  // initialize the arrays
  init(A);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  BENCHMARK_OMP(conv2D_OMP(A, B_OMP));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  BENCHMARK_CPU(conv2D(A, B));
#endif

// if test mode enabled, compare the results
#ifdef RUN_TEST
  fail += compareResults(B, B_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  free(A);
  free(B);
  free(B_OMP);

  return fail;
}
