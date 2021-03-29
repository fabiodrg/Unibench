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

def parse_kernel_log(filepath):
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



def parse_all_kernel_logs(path):
    # dict for storing the times per kernel sizes
    times = dict()
    
    for s in KERNEL_SIZES:
        times[s] = {
            'cpu': parse_kernel_log(f'{path}/{LOG_CPU_PREFIX}_{s}.log'),
            'omp_cpu': parse_kernel_log(f'{path}/{LOG_OMP_CPU_PREFIX}_{s}.log'),
            'omp_gpu': parse_kernel_log(f'{path}/{LOG_OMP_GPU_PREFIX}_{s}.log'),
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
        cols = itertools.product(KERNEL_SIZES, ('cpu', 'omp_cpu', 'omp_gpu'))
        
        # write the header
        header = ['kernel_name'] + [f'{target}_{size}' for size, target in cols]
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