/**
 * 3mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
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
#define ERROR_THRESHOLD 0.05

/* Problem size. */
#ifndef SIZE
#define SIZE 1024
#endif

#define NI SIZE
#define NJ SIZE
#define NK SIZE
#define NL SIZE
#define NM SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NM; j++) {
      C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
    }
  }

  for (i = 0; i < NM; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

int compareResults(DATA_TYPE *G, DATA_TYPE *G_OMP) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      if (percentDiff(G[i * NL + j], G_OMP[i * NL + j]) >
          ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  // printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
  //        "Percent: %d\n",
  //        ERROR_THRESHOLD, fail);
  return fail;
}

void mm3(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G) {
  int i, j, k;

  /* E := A*B */
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      E[i * NJ + j] = 0;
      for (k = 0; k < NK; ++k) {
        E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

  /* F := C*D */
  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NL; j++) {
      F[i * NL + j] = 0;
      for (k = 0; k < NM; ++k) {
        F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
      }
    }
  }

  /* G := E*F */
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      G[i * NL + j] = 0;
      for (k = 0; k < NJ; ++k) {
        G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
      }
    }
  }
}

void mm3_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G) {

/* E := A*B */
#pragma omp target teams \
  map(to : A[ : NI *NK], B[ : NK *NJ], C[ : NJ *NM], D[ : NM *NL]) \
  map(from : E[ : NI *NJ], F[ : NJ *NL], G[ : NI *NL]) \
  device(OMP_DEVICE_ID) \
  thread_limit(128)
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NJ; j++) {
        LLVM_MCA_BEGIN("kernel");
        E[i * NJ + j] = 0;
        for (int k = 0; k < NK; ++k) {
          E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        }
        LLVM_MCA_END("kernel");
      }
    }

    /* F := C*D */
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NJ; i++) {
      for (int j = 0; j < NL; j++) {
        F[i * NL + j] = 0;
        for (int k = 0; k < NM; ++k) {
          F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
        }
      }
    }

    /* G := E*F */
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NL; j++) {
        G[i * NL + j] = 0;
        for (int k = 0; k < NJ; ++k) {
          G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *F;
  DATA_TYPE *G;
  DATA_TYPE *E_OMP;
  DATA_TYPE *F_OMP;
  DATA_TYPE *G_OMP;

  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NJ * NM * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NM * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  F = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  G = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));
  E_OMP = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  F_OMP = (DATA_TYPE *)calloc(NJ * NL, sizeof(DATA_TYPE));
  G_OMP = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));

  fprintf(
      stdout,
      "<< Linear Algebra: 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) >>\n");

  init_array(A, B, C, D);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  BENCHMARK_OMP(mm3_OMP(A, B, C, D, E_OMP, F_OMP, G_OMP));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  BENCHMARK_CPU(mm3(A, B, C, D, E, F, G));
#endif

#ifdef RUN_TEST
  fail = compareResults(G, G_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(E_OMP);
  free(F);
  free(F_OMP);
  free(G);
  free(G_OMP);

  return fail;
}
