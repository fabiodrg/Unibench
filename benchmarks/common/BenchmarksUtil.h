// BenchmarksUtil.h
// Scott Grauer-Gray (sgrauerg@gmail.com)
// Functions used across hmpp codes

#ifndef BENCHMARKS_UTIL_H
#define BENCHMARKS_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/** Set default number of runs per kernel/device */
#ifndef N_RUNS
#define N_RUNS 1
#endif

/** Test mode enabled, set the defaults */
#ifdef RUN_TEST
  #if ! defined (RUN_OMP_GPU) && ! defined (RUN_OMP_CPU)
  #error "You must select one OMP version to test: RUN_OMP_GPU, RUN_OMP_CPU"
  #endif

  #define RUN_CPU_SEQ
  #define N_RUNS 1
  #define SIZE 512
#endif

/** Check if at least one device is selected */
#if ! defined(RUN_CPU_SEQ) && ! defined(RUN_OMP_GPU) && ! defined(RUN_OMP_CPU)
#error "Select one target device! Options: RUN_CPU_SEQ, RUN_OMP_GPU, RUN_OMP_CPU"
#endif

/** Check if multiple OMP versions are selected, which is not supported */
#if (defined(RUN_OMP_GPU) && defined(RUN_OMP_CPU))
#error "Multiple OMP versions are enabled, but only one OMP version will run (default: GPU)"
#endif

/** Set the OMP device ID, accordingly with selected version (if any) */
#if defined(RUN_OMP_GPU)
#define OMP_DEVICE_ID 0
#elif defined(RUN_OMP_CPU)
#define OMP_DEVICE_ID 1
#else
#define OMP_DEVICE_ID 1
#endif

/** Utility macros to run kernels N_RUNS times, collect the time, and print it */
#define __BENCHMARK(DEVICE, FUNC_CALL)                                         \
  {                                                                            \
    double __t_start, __t_end, __t_total = 0;                                  \
    for (size_t i = 0; i < N_RUNS; i++) {                                      \
      __t_start = rtclock();                                                   \
      FUNC_CALL;                                                               \
      __t_end = rtclock();                                                     \
      fprintf(stdout, DEVICE " Runtime (%lu): %0.6lfs\n", i,                    \
              __t_end - __t_start);                                            \
      __t_total += __t_end - __t_start;                                        \
    }                                                                          \
    fprintf(stdout, DEVICE " Runtime (avg): %0.6lfs\n", __t_total / N_RUNS);   \
  }

#if defined(RUN_OMP_GPU)
#define BENCHMARK_OMP(FUNC_CALL) __BENCHMARK("OMP GPU", FUNC_CALL)
#elif defined(RUN_OMP_CPU)
#define BENCHMARK_OMP(FUNC_CALL) __BENCHMARK("OMP CPU", FUNC_CALL)
#endif

#define BENCHMARK_CPU(FUNC_CALL) __BENCHMARK("CPU", FUNC_CALL)

#ifdef LLVM_MCA
#define LLVM_MCA_BEGIN(name) __asm volatile("# LLVM-MCA-BEGIN " name)
#define LLVM_MCA_END(name) __asm volatile("# LLVM-MCA-END " name)
#else
#define LLVM_MCA_BEGIN(name) 
#define LLVM_MCA_END(name) 
#endif

// define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a) {
  if (a < 0) {
    return (a * -1);
  } else {
    return a;
  }
}

float percentDiff(double val1, double val2) {
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) {
    return 0.0f;
  }

  else {
    return 100.0f *
           (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
}

#endif // BENCHMARKS_UTIL_H
