#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "main.h"



#define min(a, b) ((a<b)?a:b)
#define max(a, b) ((a>b)?a:b)

void parse(int argc, char* argv[], struct user_parameters* params)
{
    int i;
    for(i=1; i<argc; i++) {
        if(!strcmp(argv[i], "-c"))
            params->check = 1;
        else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("----------------------------------------------\n");
            printf("-                KaStORS                     -\n");
            printf("-   Kaapi Starpu OpenMP Runtime task Suite   -\n");
            printf("----------------------------------------------\n");
            printf("-h, --help : Show help information\n");
            printf("-c : Ask to check result\n");
            printf("-i : Number of iterations\n");

            printf("-n : Matrix size\n");

            printf("-b : Block size\n");
            printf("-ib : Internal Block size\n");
	    printf("-t : Choose algorithm (leavng blank will run type 1)\n(Options for type) 1 - dgeqrf task, 2 - dgetrf task, 3 - dpotrf task\n");

            exit(EXIT_SUCCESS);
        } else if(!strcmp(argv[i], "-i")) {
            if (++i < argc)
                params->niter = atoi(argv[i]);
            else {
                fprintf(stderr, "-i requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-n")) {
            if (++i < argc)
                params->matrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-n requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-b")) {
            if (++i < argc)
                params->blocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-b requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-ib")) {
            if (++i < argc)
                params->iblocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-ib requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-t")) {
            if (++i < argc)
                params->type = atoi(argv[i]);
            else {
                fprintf(stderr, "-t requires a number\n");
                exit(EXIT_FAILURE);
            }

        } else
            fprintf(stderr, "Unknown parameter : %s\n", argv[i]);
    }
}

int comp (const void * elem1, const void * elem2)
{
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

int main(int argc, char* argv[])
{
    int num_threads = 1;
    struct user_parameters params;
    memset(&params, 0, sizeof(params));

    /* default value */
    params.niter = 1;

    parse(argc, argv, &params);

// get Number of thread if OpenMP is activated
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads();
#endif

    // warmup
    run(&params);

    double mean = 0.0;
    double meansqr = 0.0;
    double min_ = DBL_MAX;
    double max_ = -1;
    double* all_times = (double*)malloc(sizeof(double) * params.niter);

    for (int i=0; i<params.niter; ++i)
    {
      double cur_time = run(&params);
      all_times[i] = cur_time;
      mean += cur_time;
      min_ = min(min_, cur_time);
      max_ = max(max_, cur_time);
      meansqr += cur_time * cur_time;
      }
    mean /= params.niter;
    meansqr /= params.niter;
    double stddev = sqrt(meansqr - mean * mean);

    qsort(all_times, params.niter, sizeof(double), comp);
    double median = all_times[params.niter / 2];

    free(all_times);

    printf("Program : %s\n", argv[0]);
    printf("Size : %d\n", params.matrix_size);


    printf("Blocksize : %d\n", params.blocksize);

    printf("Internal Blocksize : %d\n", params.iblocksize);

    printf("Iterations : %d\n", params.niter);

    printf("Threads : %d\n", num_threads);
    printf("Gflops:: ");

    printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
           mean, stddev, min_, max_, median);
    if(params.check)
        printf("Check : %s\n", (params.succeed)?
                ((params.succeed > 1)?"not implemented":"success")
                :"fail");
    if (params.string2display !=0)
      printf("%s", params.string2display);
    printf("\n");

    return 0;
}
