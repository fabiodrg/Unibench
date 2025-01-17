#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void lud_omp_cpu(float *a, int size) {
  int i, j, k;
  float sum;

  for (i = 0; i < size; i++) {
    for (j = i; j < size; j++) {
      sum = a[i * size + j];
      for (k = 0; k < i; k++)
        sum -= a[i * size + k] * a[k * size + j];
      a[i * size + j] = sum;
    }

    for (j = i + 1; j < size; j++) {
      sum = a[j * size + i];
      for (k = 0; k < i; k++)
        sum -= a[j * size + k] * a[k * size + i];
      a[j * size + i] = sum / a[i * size + i];
    }
  }
}

void lud_omp_gpu(float *a, int size) {
  int i, j, k;
  float sum;

  #pragma omp target data map(tofrom : a[0 : size *size])
  {
    for (i = 0; i < size; i++) {
      #pragma omp target teams distribute parallel for
      for (j = i; j < size; j++) {
        sum = a[i * size + j];
        for (k = 0; k < i; k++)
          sum -= a[i * size + k] * a[k * size + j];
        a[i * size + j] = sum;
      }

      #pragma omp target teams distribute parallel for
      for (j = i + 1; j < size; j++) {
        sum = a[j * size + i];
        for (k = 0; k < i; k++)
          sum -= a[j * size + k] * a[k * size + i];
        a[j * size + i] = sum / a[i * size + i];
      }
    }
  }
}
