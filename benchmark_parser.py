from os import scandir, path
import pprint
import re
import csv
import itertools

LOGS_DIR="./logs/Polybench"
LOG_CPU_PREFIX="cpu"
LOG_OMP_CPU_PREFIX="omp_cpu"
LOG_OMP_GPU_PREFIX="omp_gpu"

KERNEL_SIZES=(512, 1024, 2048, 4096)
KERNEL_SIZES_NAMES = {
    512: 'Mini',
    1024: 'Small',
    2048: 'Medium',
    4096: 'Large',
}

# number of decimal digits to represent time measurements
FLOAT_DIGITS=6

# formats floating point numbers as strings, with the number of decimal digits
# set in 'FLOAT_DIGITS'
format_float = lambda f: format(f, f'.{FLOAT_DIGITS}f')


def get_speedup(cpu_t:float, acc_t:float) -> float:
    """Computes the speedup in offloading to accelerator compared to sequential
    CPU
    
    Formula: cpu_t/acc_t
    If the execution in the accelerator is slower, it returns a negative
    value. E.g. cpu_t=1, acc_t=0.5, it returns -2 to denote execution on
    accelerator is two times slower
    
    Args:
        cpu_t: Time in seconds for sequential CPU execution
        acc_t: Time in seconds for accelerator device execution
    """
    if (cpu_t == None or acc_t == None):
        return None
    
    if acc_t > cpu_t:
        return -acc_t/cpu_t
    else:
        return cpu_t/acc_t

def parse_kernel_log(filepath):
    """Parses a log file with execution time measurements and returns the
    average

    Args:
        filepath: The path to the log file (str)
    
    Returns:
        The average time or None if no data is available or parsing failed
    """
    try:
        with open(filepath, "r") as f:
            # parse all measured times
            times = re.findall(r'Runtime \(\d+\): (\d+.\d+)', f.read())
            # convert time values to floats
            times = [float(t) for t in times]
            # return the average time if non-empty list
            if times:
                return sum(times)/len(times)
            else:
                return None
    except:
        print(f'[DEBUG] Failed to open {filepath}')
        return None

def parse_all_kernel_logs(path:str) -> dict:
    """Parses all logs with respect to a kernel, located on the same parent dir

    Given a path, e.g. ./logs/Polybench/2MM, it iterates over the logs with
    filenames `<target prefix>_<size>.log`, parses the execution times and
    stores the average on a dictionary. It adds the speedups relative to CPU
    sequential runtime.

    Args:
        path: The parent directory for a kenel's log files
    
    Returns: Nested dictionary with average execution times and speedup values
    for different problem sizes
    """
    # dict for storing the times per kernel sizes
    times = dict()
    my_format = lambda fp_num: format_float(fp_num) if fp_num else "nan"
    for s in KERNEL_SIZES:
        cpu_t = parse_kernel_log(f'{path}/{LOG_CPU_PREFIX}_{s}.log')
        omp_cpu_t = parse_kernel_log(f'{path}/{LOG_OMP_CPU_PREFIX}_{s}.log')
        omp_gpu_t = parse_kernel_log(f'{path}/{LOG_OMP_GPU_PREFIX}_{s}.log')
        speedup_omp_cpu = get_speedup(cpu_t, omp_cpu_t)
        speedup_omp_gpu = get_speedup(cpu_t, omp_gpu_t)
        
        times[s] = {
            'cpu': my_format(cpu_t),
            'omp_cpu': my_format(omp_cpu_t),
            'omp_gpu': my_format(omp_gpu_t),
            'speedup_omp_cpu':  my_format(speedup_omp_cpu),
            'speedup_omp_gpu': my_format(speedup_omp_gpu),
        }
    
    return times

def parse_logs():
    times = dict()
    for kernel_dir in scandir(LOGS_DIR):
        if kernel_dir.is_dir():
            times[kernel_dir.name] = parse_all_kernel_logs(kernel_dir.path)
    
    return times

def dump_csv(times):
    with open('polybench.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')

        # product of kernel sizes and type of target device
        cols = itertools.product(KERNEL_SIZES, ('cpu', 'ompCpu', 'ompGpu', 'speedOmpCpu', 'speedOmpGpu'))
        
        # write the header
        header = ['kernelName'] + [f'{target}{KERNEL_SIZES_NAMES[size]}' for size, target in cols]
        writer.writerow(header)

        # for each kernel, dump the timings
        for kernel, measures in times.items():
            # chain all measures for each kernel size, creating a plain list of measures in seconds
            # and respecting the header ordering: cpu_512, omp_cpu_512, omp_gpu_512, cpu_1024, ...
            t = list(itertools.chain.from_iterable([_.values() for _ in measures.values()]))
            # write the row (kernel name followed by times)
            writer.writerow([kernel] + t)



times = parse_logs()
dump_csv(times)