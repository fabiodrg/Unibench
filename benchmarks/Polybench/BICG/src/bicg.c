/**
 * bicg.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
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
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

// Error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.7

/* Problem size. */
#ifndef SIZE
#define SIZE 1024
#endif

#define NX SIZE
#define NY SIZE

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r) {
  int i, j;

  for (i = 0; i < NX; i++) {
    r[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
    }
  }

  for (i = 0; i < NY; i++) {
    p[i] = i * M_PI;
  }
}

int compareResults(DATA_TYPE *s, DATA_TYPE *s_outputFromGpu, DATA_TYPE *q,
                   DATA_TYPE *q_outputFromGpu) {
  int i, fail;
  fail = 0;

  // Compare s with s_cuda
  for (i = 0; i < NX; i++) {
    if (percentDiff(q[i], q_outputFromGpu[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  for (i = 0; i < NY; i++) {
    if (percentDiff(s[i], s_outputFromGpu[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);

  return fail;
}

void bicg(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
              DATA_TYPE *q) {
  int i, j;

  for (i = 0; i < NY; i++) {
    s[i] = 0.0;
  }

  for (i = 0; i < NX; i++) {
    q[i] = 0.0;
    for (j = 0; j < NY; j++) {
      s[j] = s[j] + r[i] * A[i * NY + j];
      q[i] = q[i] + A[i * NY + j] * p[j];
    }
  }
}

void bicg_OMP(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
              DATA_TYPE *q) {
  int i, j;

  for (i = 0; i < NY; i++) {
    s[i] = 0.0;
  }

  #pragma omp target teams map(to : A[ : NX *NY], p[ : NY], r[ : NX]) map(tofrom : s[ : NY], q[ : NX]) device(OMP_DEVICE_ID)
  {
    #pragma omp distribute parallel for private(i)
    for (j = 0; j < NY; j++) {
      LLVM_MCA_BEGIN("kernel");
      for (i = 0; i < NX; i++) {
        s[j] = s[j] + r[i] * A[i * NY + j];
      }
      LLVM_MCA_END("kernel");
    }

    #pragma omp distribute parallel for private(j)
    for (i = 0; i < NX; i++) {
      q[i] = 0.0;
      for (j = 0; j < NY; j++) {
        q[i] = q[i] + A[i * NY + j] * p[j];
      }
    }
  }
}

int main(int argc, char **argv) {
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *r;
  DATA_TYPE *s;
  DATA_TYPE *p;
  DATA_TYPE *q;
  DATA_TYPE *s_OMP;
  DATA_TYPE *q_OMP;

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  r = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  p = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  q = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s_OMP = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  q_OMP = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< BiCG Sub Kernel of BiCGStab Linear Solver >>\n");

  init_array(A, p, r);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  BENCHMARK_OMP(bicg_OMP(A, r, s_OMP, p, q_OMP));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  BENCHMARK_CPU(bicg(A, r, s, p, q));
#endif

#ifdef RUN_TEST
  fail = compareResults(s, s_OMP, q, q_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_OMP);
  free(q_OMP);

  return fail;
}
