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
#ifndef IN_RUNS
#define IN_RUNS 1
#endif

/** Test mode enabled, set the defaults */
#ifdef RUN_TEST
#if !defined(RUN_OMP_GPU) && !defined(RUN_OMP_CPU)
#error "You must select one OMP version to test: RUN_OMP_GPU, RUN_OMP_CPU"
#endif

#define RUN_CPU_SEQ
#define IN_RUNS 1
#define SIZE 1024
#endif

/** Pre-set problem sizes */
#ifdef _OPS_O2
#define _SIZE_FACTOR 5
#else
#define _SIZE_FACTOR 1
#endif

#ifdef MINI
#define SIZE 512*_SIZE_FACTOR
#elif SMALL
#define SIZE 1024*_SIZE_FACTOR
#elif MEDIUM
#define SIZE 2048*_SIZE_FACTOR
#elif LARGE
#define SIZE 4096*_SIZE_FACTOR
#endif

/** Set default problem size, if undefined */
#ifndef SIZE
#error "No problem size set!"
#endif

/** Check if at least one device is selected */
#if !defined(RUN_CPU_SEQ) && !defined(RUN_OMP_GPU) && !defined(RUN_OMP_CPU)
#error                                                                         \
    "Select one target device! Options: RUN_CPU_SEQ, RUN_OMP_GPU, RUN_OMP_CPU"
#endif

/** Check if multiple OMP versions are selected, which is not supported */
#if (defined(RUN_OMP_GPU) && defined(RUN_OMP_CPU))
#error                                                                         \
    "Multiple OMP versions are enabled, but only one OMP version will run (default: GPU)"
#endif

/** Set the OMP device ID, accordingly with selected version (if any) */
#if defined(RUN_OMP_GPU)
#define OMP_DEVICE_ID 0
#elif defined(RUN_OMP_CPU)
#define OMP_DEVICE_ID 1
#else
#define OMP_DEVICE_ID 1
#endif

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.5

/**
 * @brief Run function responsible to setup everything before running the actual
 * kernel.
 *
 * The function is executed IN_RUNS times. Each call repeats the memory
 * allocation and initialization process before calling the kernel, as an
 * attempt to ensure cold cache
 *
 */
#define BENCH_INIT(KERNEL_SETUP_FUNC)                                          \
  {                                                                            \
    for (size_t i = 0; i < IN_RUNS; i++) {                                     \
      KERNEL_SETUP_FUNC();                                                     \
    }                                                                          \
  }

/**
 * @brief Measures the execution time to execute the kernels function and
 * outputs to stdout
 *
 */
#define __BENCHMARK(DEVICE, FUNC_CALL)                                         \
  {                                                                            \
    double __t_start, __t_end;                                                 \
    __t_start = rtclock();                                                     \
    FUNC_CALL;                                                                 \
    __t_end = rtclock();                                                       \
    fprintf(stdout, DEVICE " Runtime: %0.6lfs\n", __t_end - __t_start);        \
  }

/**
 * @brief Auxiliar macro to launch OMP related benchmarks
 * @see __BENCHMARK
 */
#if defined(RUN_OMP_GPU)
#define BENCHMARK_OMP(FUNC_CALL) __BENCHMARK("OMP GPU", FUNC_CALL)
#elif defined(RUN_OMP_CPU)
#define BENCHMARK_OMP(FUNC_CALL) __BENCHMARK("OMP CPU", FUNC_CALL)
#endif

/**
 * @brief Auxiliar macro to launch CPU sequential benchmarks
 * @see __BENCHMARK
 */
#define BENCHMARK_CPU(FUNC_CALL) __BENCHMARK("CPU", FUNC_CALL)

#ifdef LLVM_MCA
#define LLVM_MCA_BEGIN(name) __asm volatile("# LLVM-MCA-BEGIN " name)
#define LLVM_MCA_END(name) __asm volatile("# LLVM-MCA-END " name)
#else
#define LLVM_MCA_BEGIN(name)
#define LLVM_MCA_END(name)
#endif

/**
 * @brief Used to prevent dead-code elimination optimizations
 * It only has to be used if a the test mode is not enabled, otherwise
 * the computation results are used for comparasion
 * @param array A pointer of type DATA_TYPE
 * @param elems The number of elements in the array
 */
#ifndef RUN_TEST
#define DCE_PREVENT(array, elems)                                              \
  {                                                                            \
    DATA_TYPE __acc = 0;                                                       \
    for (size_t __i = 0; __i < elems; __i++)                                   \
      __acc += array[__i];                                                     \
    fprintf(stderr, "dead code elimination prevent: acc = %lf\n", __acc);      \
  }
#else
#define DCE_PREVENT(array, elems)
#endif

#define SALUTE(msg) fprintf(stdout, ">> %s (N=%d) <<\n", msg, SIZE)

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

#endif
