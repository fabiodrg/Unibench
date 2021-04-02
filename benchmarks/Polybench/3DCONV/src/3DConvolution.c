/**
 * 3DConvolution.c: This file was adapted from PolyBench/GPU 1.0 test suite
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
#define ERROR_THRESHOLD 0.5

#ifndef SIZE
#define SIZE 1024
#endif

#define NI SIZE
#define NJ SIZE
#define NK SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv3D(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

  for (j = 1; j < NJ - 1; ++j) {
    for (i = 1; i < NI - 1; ++i) {
      for (k = 1; k < NK - 1; ++k) {
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
  }
}

void conv3D_OMP(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

#pragma omp target teams distribute parallel for \
  map(to: A[:NI * NJ * NK])       \
  map(from: B[:NI * NJ * NK]) \
  device(OMP_DEVICE_ID) \
  private(i, k)
  for (j = 1; j < NJ - 1; ++j) {
    LLVM_MCA_BEGIN("kernel");
    for (i = 1; i < NI - 1; ++i) {
      for (k = 1; k < NK - 1; ++k) {
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
    LLVM_MCA_END("kernel");
  }
}

void init(DATA_TYPE *A) {
  int i, j, k;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      for (k = 0; k < NK; ++k) {
        A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
      }
    }
  }
}

int compareResults(DATA_TYPE *B, DATA_TYPE *B_OMP) {
  int i, j, k, fail;
  fail = 0;

  // Compare result from cpu and gpu...
  for (i = 1; i < NI - 1; ++i) {
    for (j = 1; j < NJ - 1; ++j) {
      for (k = 1; k < NK - 1; ++k) {
        if (percentDiff(B[i * (NK * NJ) + j * NK + k],
                        B_OMP[i * (NK * NJ) + j * NK + k]) > ERROR_THRESHOLD) {
          fail++;
        }
      }
    }
  }

  // Print results
  // printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
  //        "Percent: %d\n",
  //        ERROR_THRESHOLD, fail);

  return fail;
}

int main(int argc, char *argv[]) {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_OMP;

  A = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Three dimensional (3D) convolution <<\n");

  init(A);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  B_OMP = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  BENCHMARK_OMP(conv3D_OMP(A, B_OMP));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  B = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  BENCHMARK_CPU(conv3D(A, B));
#endif

// if test mode enabled, compare the results
#ifdef RUN_TEST
  fail += compareResults(B, B_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

// Release memory
  free(A);

#ifdef RUN_CPU_SEQ
  free(B);
#endif

#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  free(B_OMP);
#endif

  return fail;
}
