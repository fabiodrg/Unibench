/**
 * atax.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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

#define NX SIZE
#define NY SIZE

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A) {
  int i, j;

  for (i = 0; i < NX; i++) {
    x[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

int compareResults(DATA_TYPE *z, DATA_TYPE *z_OMP) {
  int i, fail;
  fail = 0;

  for (i = 0; i < NY; i++) {
    if (percentDiff(z[i], z_OMP[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  return fail;
}

void atax(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  int i, j;

  for (i = 0; i < NY; i++) {
    y[i] = 0;
  }

  for (i = 0; i < NX; i++) {
    tmp[i] = 0;

    for (j = 0; j < NY; j++) {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }

    for (j = 0; j < NY; j++) {
      y[j] = y[j] + A[i * NY + j] * tmp[i];
    }
  }
}

void atax_OMP(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {

  for (int i = 0; i < NY; i++) {
    y[i] = 0;
  }

  #pragma omp target teams map(to : A[ : NX *NY], x[ : NY]) map(tofrom : tmp[ : NX], y[ : NY]) device(OMP_DEVICE_ID)
  {
    #pragma omp distribute parallel for
    for (int i = 0; i < NX; i++) {
      LLVM_MCA_BEGIN("kernel");
      tmp[i] = 0;
      for (int j = 0; j < NY; j++) {
        tmp[i] += A[i * NY + j] * x[j];
      }
      LLVM_MCA_END("kernel");
    }

    // Note that the Loop has been reversed
    #pragma omp distribute parallel for
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {
        y[j] += A[i * NY + j] * tmp[i];
      }
    }
  }
}

int main(int argc, char **argv) {
  fprintf(stdout, "<< Matrix Transpose and Vector Multiplication >>\n");

  // declare arrays and allocate memory
  DATA_TYPE *A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  DATA_TYPE *x = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  DATA_TYPE *y = NULL;
  DATA_TYPE *y_OMP = NULL;
  DATA_TYPE *tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  // initialize arrays
  init_array(x, A);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  y_OMP = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  BENCHMARK_OMP(atax_OMP(A, x, y_OMP, tmp));
  // prevent dead code elimination
  DCE_PREVENT(y_OMP, NY);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  y = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  BENCHMARK_CPU(atax(A, x, y, tmp));
  // prevent dead code elimination
  DCE_PREVENT(y, NY);
#endif

  int fail = 0;
// if test mode enabled, compare the results
#ifdef RUN_TEST
  fail = compareResults(y, y_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // Release memory
  free(A);
  free(x);
  free(y);
  free(y_OMP);
  free(tmp);

  return fail;
}
