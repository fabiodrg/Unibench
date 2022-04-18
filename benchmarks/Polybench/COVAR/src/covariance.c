/**
 * covariance.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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

/* Problem size */
#define M SIZE
#define N SIZE

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *data) {
  int i, j;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
    }
  }
}

int compareResults(DATA_TYPE *symmat, DATA_TYPE *symmat_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      if (percentDiff(symmat[i * (N + 1) + j],
                      symmat_outputFromGpu[i * (N + 1) + j]) >
          ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  return fail;
}

void covariance(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean) {
  int i, j, j1, j2;

  /* Determine mean of column vectors of input data matrix */
  for (j = 1; j < (M + 1); j++) {
    mean[j] = 0.0;
    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }
    mean[j] /= FLOAT_N;
  }

  /* Center the column vectors. */
  for (i = 1; i < (N + 1); i++) {
    for (j = 1; j < (M + 1); j++) {
      data[i * (M + 1) + j] -= mean[j];
    }
  }

  /* Calculate the m * m covariance matrix. */
  for (j1 = 1; j1 < (M + 1); j1++) {
    for (j2 = j1; j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;
      for (i = 1; i < N + 1; i++) {
        symmat[j1 * (M + 1) + j2] +=
            data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
      }
      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }
}

void covariance_OMP(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean) {

  /* Determine mean of column vectors of input data matrix */

  #pragma omp target data \
    map(to: data[:(M + 1) * (N + 1)]) \
    map(alloc: mean[:(M + 1)]) \
    map(tofrom: symmat[:(M + 1) * (N + 1)]) device(OMP_DEVICE_ID)
  {
    #pragma omp target teams distribute parallel for device(OMP_DEVICE_ID)
    for (int j = 1; j < (M + 1); j++) {
      mean[j] = 0.0;
      for (int i = 1; i < (N + 1); i++) {
        mean[j] += data[i * (M + 1) + j];
      }
      mean[j] /= FLOAT_N;
    }

    /* Center the column vectors. */
    #pragma omp target teams distribute parallel for collapse(2) device(OMP_DEVICE_ID)
    for (int i = 1; i < (N + 1); i++) {
      for (int j = 1; j < (M + 1); j++) {
        data[i * (M + 1) + j] -= mean[j];
      }
    }

    /* Calculate the m * m covariance matrix. */
    #pragma omp target teams distribute parallel for device(OMP_DEVICE_ID)
    for (int j1 = 1; j1 < (M + 1); j1++) {
      for (int j2 = j1; j2 < (M + 1); j2++) {
        symmat[j1 * (M + 1) + j2] = 0.0;
        for (int i = 1; i < N + 1; i++) {
          symmat[j1 * (M + 1) + j2] +=
              data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
        }
        symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
      }
    }
  }
}

int main() {
  SALUTE("Covariance Computation");

  // declare arrays and allocate common memory
  DATA_TYPE *data = NULL;
  DATA_TYPE *data_OMP = NULL;
  DATA_TYPE *symmat = NULL;
  DATA_TYPE *symmat_OMP = NULL;
  DATA_TYPE *mean = (DATA_TYPE *)calloc((M + 1), sizeof(DATA_TYPE));

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  symmat_OMP = (DATA_TYPE *)calloc((M + 1) * (M + 1), sizeof(DATA_TYPE));
  data_OMP = (DATA_TYPE *)calloc((M + 1) * (N + 1), sizeof(DATA_TYPE));
  init_arrays(data_OMP);
  BENCHMARK_OMP(covariance_OMP(data_OMP, symmat_OMP, mean));
  // prevent dead-code elimination
  DCE_PREVENT(symmat_OMP, (M+1)*(M+1));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  symmat = (DATA_TYPE *)calloc((M + 1) * (M + 1), sizeof(DATA_TYPE));
  data = (DATA_TYPE *)calloc((M + 1) * (N + 1), sizeof(DATA_TYPE));
  init_arrays(data);
  BENCHMARK_CPU(covariance(data, symmat, mean));
  // prevent dead-code elimination
  DCE_PREVENT(symmat, (M+1)*(M+1));
#endif

  int fail = 0;
// if TEST is enabled, then compare OMP results against sequential mode
#ifdef RUN_TEST
  fail = compareResults(symmat, symmat_OMP);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // release memory
  free(data);
  free(data_OMP);
  free(symmat);
  free(symmat_OMP);
  free(mean);

  return fail;
}
