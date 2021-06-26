/**
 * correlation.c This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *  	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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

/* Problem size */
#define M SIZE
#define N SIZE

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

/**
 * @brief Initialize matrice
 * 
 * @param data 
 */
void init_arrays(DATA_TYPE *data) {
  int i, j;

  for (i = 0; i < (M + 1); i++) {
    for (j = 0; j < (N + 1); j++) {
      data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / (M + 1);
    }
  }
}

/**
 * @brief 
 * 
 * @param symmat 
 * @param symmat_outputFromGpu 
 * @return int 
 */
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


/**
 * @brief 
 * 
 * @param data 
 * @param mean 
 * @param stddev 
 * @param symmat 
 */
void correlation(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev,
                 DATA_TYPE *symmat) {
  int i, j, j1, j2;

  // Determine mean of column vectors of input data matrix
  for (j = 1; j < (M + 1); j++) {
    mean[j] = 0.0;

    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }

    mean[j] /= (DATA_TYPE)FLOAT_N;
  }

  // Determine standard deviations of column vectors of data matrix.
  for (j = 1; j < (M + 1); j++) {
    stddev[j] = 0.0;

    for (i = 1; i < (N + 1); i++) {
      stddev[j] +=
          (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
  }

  // i - threadIdx.x, j = threadIdx.y
  // Center and reduce the column vectors.
  for (i = 1; i < (N + 1); i++) {
    for (j = 1; j < (M + 1); j++) {
      data[i * (M + 1) + j] -= mean[j];
      data[i * (M + 1) + j] /= (sqrt(FLOAT_N) * stddev[j]);
    }
  }

  // Calculate the m * m correlation matrix.
  for (j1 = 1; j1 < M; j1++) {
    symmat[j1 * (M + 1) + j1] = 1.0;

    for (j2 = j1 + 1; j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;

      for (i = 1; i < (N + 1); i++) {
        symmat[j1 * (M + 1) + j2] +=
            (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
      }

      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
}

/**
 * @brief 
 * 
 * @param data 
 * @param mean 
 * @param stddev 
 * @param symmat 
 */
void correlation_OMP(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev,
                     DATA_TYPE *symmat) {
  int i, j, k;
  #pragma omp target data map(to: data[:(M+1)*(N+1)], mean[:(M+1)], stddev[:(M+1)]) map(tofrom: symmat[:(M+1)*(N+1)]) device(OMP_DEVICE_ID)
  {
    // Determine mean of column vectors of input data matrix
    #pragma omp target teams distribute  parallel for private(i) device(OMP_DEVICE_ID)
    for (j = 1; j < (M + 1); j++) {
      mean[j] = 0.0;
      for (i = 1; i < (N + 1); i++) {
        mean[j] += data[i * (M + 1) + j];
      }
      mean[j] /= (DATA_TYPE)FLOAT_N;
    }

    // Determine standard deviations of column vectors of data matrix.
    #pragma omp target teams distribute parallel for private(i) device(OMP_DEVICE_ID)
    for (j = 1; j < (M + 1); j++) {
      stddev[j] = 0.0;
      for (i = 1; i < (N + 1); i++) {
        stddev[j] +=
          (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
      }

      stddev[j] /= FLOAT_N;
      stddev[j] = sqrt(stddev[j]);
      if (stddev[j] <= EPS) {
        stddev[j] = 1.0;
      }
    }

    // Center and reduce the column vectors.
    #pragma omp target teams distribute parallel for collapse(2) device(OMP_DEVICE_ID)
    for (i = 1; i < (N + 1); i++) {
      for (j = 1; j < (M + 1); j++) {
        data[i * (M + 1) + j] -= mean[j];
        data[i * (M + 1) + j] /= (sqrt(FLOAT_N) * stddev[j]);
      }
    }

    // Calculate the m * m correlation matrix.
    #pragma omp target teams distribute parallel for private(j, i) device(OMP_DEVICE_ID)
    for (k = 1; k < M; k++) {
      symmat[k * (M + 1) + k] = 1.0;
      for (j = k + 1; j < (M + 1); j++) {
        symmat[k * (M + 1) + j] = 0.0;
        for (i = 1; i < (N + 1); i++) {
          symmat[k * (M + 1) + j] +=
            (data[i * (M + 1) + k] * data[i * (M + 1) + j]);
        }
        symmat[j * (M + 1) + k] = symmat[k * (M + 1) + j];
      }
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
}

int main() {
  fprintf(stdout, "<< Correlation Computation >>\n");

  // declare arrays and allocate memory
  DATA_TYPE *data = (DATA_TYPE *) malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  DATA_TYPE *mean = (DATA_TYPE *) malloc((M + 1) * sizeof(DATA_TYPE));
  DATA_TYPE *stddev = (DATA_TYPE *) malloc((M + 1) * sizeof(DATA_TYPE));
  DATA_TYPE *symmat = NULL;
  DATA_TYPE *symmat_GPU = NULL;

  // init operand matrices
  init_arrays(data);

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  symmat_GPU = (DATA_TYPE *) malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  BENCHMARK_OMP(correlation_OMP(data, mean, stddev, symmat_GPU));
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  symmat = (DATA_TYPE *) malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  BENCHMARK_CPU(correlation(data, mean, stddev, symmat));
#endif

  int fail = 0;
// if TEST is enabled, then compare OMP results against sequential mode
#ifdef RUN_TEST
  fail = compareResults(symmat, symmat_GPU);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // release memory
  free(data);
  free(mean);
  free(stddev);
  free(symmat_GPU);
  free(symmat);

  return fail;
}
