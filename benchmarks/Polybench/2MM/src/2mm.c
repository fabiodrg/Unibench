/**
 * 2mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
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

#define NI SIZE
#define NJ SIZE
#define NK SIZE
#define NL SIZE

#pragma GCC diagnostic ignored "-Wmacro-redefined"
#define ERROR_THRESHOLD 1.5

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

/**
 * @brief Initialize operand matrices
 *
 * @param A
 * @param B
 * @param D
 */
void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

/**
 * @brief 
 * 
 * @param E Expected result matrix 
 * @param E_OMP Obtained matrix
 * @return int Number of detected fails
 */
int compareResults(DATA_TYPE *E, DATA_TYPE *E_OMP) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NI; j++) {
      if (percentDiff(E[i * NI + j], E_OMP[i * NI + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  
  return fail;
}

/**
 * @brief Sequential CPU version to compute A.B.D matrixes
 *
 * @param A Input
 * @param B Input
 * @param C Auxiliar
 * @param D Input
 * @param E Output
 */
void mm2(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = 0.0;
      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      E[i * NL + j] = 0.0;
      for (k = 0; k < NJ; ++k) {
        E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
      }
    }
  }
}

/**
 * @brief OMP version to compute A.B.D matrixes
 *
 * @param A Input
 * @param B Input
 * @param C Auxiliar
 * @param D Input
 * @param E Output
 */
void mm2_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E) {

#pragma omp target teams map(from: E[:NI*NL], C[:NI*NJ]) map(to: A[:NI*NK], B[:NK*NJ], D[:NJ*NL]) device(OMP_DEVICE_ID) 
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NJ; j++) {
        LLVM_MCA_BEGIN("kernel");
        C[i * NJ + j] = 0.0;
        for (int k = 0; k < NK; ++k) {
          C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        }
        LLVM_MCA_END("kernel");
      }
    }

    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NL; j++) {
        E[i * NL + j] = 0.0;
        for (int k = 0; k < NJ; ++k) {
          E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  fprintf(stdout,
          "<< Linear Algebra: 2 Matrix Multiplications (C=A.B; E=C.D) >>\n");

  // declare arrays and allocate memory
  DATA_TYPE *A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  DATA_TYPE *D = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  DATA_TYPE *C = NULL;
  DATA_TYPE *C_OMP = NULL;
  DATA_TYPE *E = NULL;
  DATA_TYPE *E_OMP = NULL;

  // init operand matrices
  init_array(A, B, D);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  C_OMP = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  E_OMP = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));
  BENCHMARK_OMP(mm2_OMP(A, B, C_OMP, D, E_OMP));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  C = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  E = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));
  BENCHMARK_CPU(mm2(A, B, C, D, E));
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail += compareResults(C, C_OMP);
  fail += compareResults(E, E_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  /** Release memory */
  free(A);
  free(B);
  free(D);
  free(C_OMP);
  free(E_OMP);
  free(C);
  free(E);

  return fail;
}
