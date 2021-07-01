/**
 * fdtd2d.c: This file was adapted from PolyBench/GPU 1.0 test suite
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
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

#define tmax 500
#define NX SIZE
#define NY SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey,
                 DATA_TYPE *hz) {
  int i, j;

  for (i = 0; i < tmax; i++) {
    _fict_[i] = (DATA_TYPE)i;
  }

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
      ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
      hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
    }
  }
}

void init_array_hz(DATA_TYPE *hz) {
  int i, j;

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
    }
  }
}

int compareResults(DATA_TYPE *hz1, DATA_TYPE *hz2) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      if (percentDiff(hz1[i * NY + j], hz2[i * NY + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  return fail;
}

void runFdtd(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz) {
  int t, i, j;

  for (t = 0; t < tmax; t++) {
    for (j = 0; j < NY; j++) {
      ey[0 * NY + j] = _fict_[t];
    }

    for (i = 1; i < NX; i++) {
      for (j = 0; j < NY; j++) {
        ey[i * NY + j] =
            ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
      }
    }

    for (i = 0; i < NX; i++) {
      for (j = 1; j < NY; j++) {
        ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] -
                               0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
      }
    }

    for (i = 0; i < NX; i++) {
      for (j = 0; j < NY; j++) {
        hz[i * NY + j] =
            hz[i * NY + j] -
            0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                   ey[(i + 1) * NY + j] - ey[i * NY + j]);
      }
    }
  }
}

void runFdtd_OMP(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey,
                 DATA_TYPE *hz) {
  int t, i, j;
  
  #pragma omp target data  map(to : _fict_[ : tmax], ex[ : (NX *(NY + 1))], ey[ : ((NX + 1) * NY)]) map(tofrom : hz[ : NX *NY]) device(OMP_DEVICE_ID)
  {
    for (t = 0; t < tmax; t++) {
      #pragma omp target teams distribute parallel for device(OMP_DEVICE_ID)
      for (j = 0; j < NY; j++) {
        ey[0 * NY + j] = _fict_[t];
      }

      #pragma omp target teams distribute parallel for collapse(2) device(OMP_DEVICE_ID)
      for (i = 1; i < NX; i++) {
        for (j = 0; j < NY; j++) {
          ey[i * NY + j] =
              ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
        }
      }

      #pragma omp target teams distribute parallel for collapse(2) device(OMP_DEVICE_ID)
      for (i = 0; i < NX; i++) {
        for (j = 1; j < NY; j++) {
          ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] -
                                 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
        }
      }
      
      #pragma omp target teams distribute parallel for collapse(2) device(OMP_DEVICE_ID)
      for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
          hz[i * NY + j] =
              hz[i * NY + j] -
              0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                     ey[(i + 1) * NY + j] - ey[i * NY + j]);
        }
      }
    }
  }
}

int main() {
  fprintf(stdout, "<< 2-D Finite Different Time Domain Kernel >>\n");

  // declare arrays and allocate memory
  DATA_TYPE *_fict_ = (DATA_TYPE *)calloc(tmax, sizeof(DATA_TYPE));
  DATA_TYPE *ex = (DATA_TYPE *)calloc(NX * (NY + 1), sizeof(DATA_TYPE));
  DATA_TYPE *ey = (DATA_TYPE *)calloc((NX + 1) * NY, sizeof(DATA_TYPE));
  DATA_TYPE *hz = NULL;
  DATA_TYPE *hz_outputFromGpu = NULL;

// run OMP on GPU or CPU if enabled
#if defined(RUN_OMP_GPU) || defined(RUN_OMP_CPU)
  // allocate
  hz_outputFromGpu = (DATA_TYPE *) malloc(NX * NY * sizeof(DATA_TYPE));
  // init arrays
  init_arrays(_fict_, ex, ey, hz_outputFromGpu);
  // benchmark
  BENCHMARK_OMP(runFdtd_OMP(_fict_, ex, ey, hz_outputFromGpu));
  // prevent dead-code elimination
  DCE_PREVENT(hz_outputFromGpu, NX*NY);
#endif

// run sequential version if enabled
#ifdef RUN_CPU_SEQ
  // reset memory for common/shared arrays
  // NOTE: it seems the init_arrays does not overwrite all memory positions
  // on ex and ey. In turn, that seems to affect the final output and generate
  // errors. Since I dont know the algorithm, this is a quick fix that resets
  // the arrays with 0s, the then initializes everything again
  memset(_fict_, 0, tmax * sizeof(DATA_TYPE));
  memset(ex, 0, NX * (NY + 1) * sizeof(DATA_TYPE));
  memset(ey, 0, (NX + 1) * NY * sizeof(DATA_TYPE));
  // allocate
  hz = (DATA_TYPE *) malloc(NX * NY * sizeof(DATA_TYPE));
  // init arrays
  init_arrays(_fict_, ex, ey, hz);
  // benchmark
  BENCHMARK_CPU(runFdtd(_fict_, ex, ey, hz));
  // prevent dead-code elimination
  DCE_PREVENT(hz, NX*NY);
#endif

  // if TEST is enabled, then compare OMP results against sequential mode
  int fail = 0;
#ifdef RUN_TEST
  fail = compareResults(hz, hz_outputFromGpu);
  printf("Errors on OMP (threshold %4.2lf): %d\n", ERROR_THRESHOLD, fail);
#endif

  // release memory
  free(_fict_);
  free(ex);
  free(ey);
  free(hz);
  free(hz_outputFromGpu);

  return fail;
}
