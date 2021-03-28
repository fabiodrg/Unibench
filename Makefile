# README
# 
# Environment variables
# BENCH_NAME: The benchmark suite name and the kernel name. Should match a directory in ./benchmarks. E.g. Polybench/2MM
# SIZE: The problem dimensions (e.g. 512, 1024, 2048, 4096)
# 

#############################################
# Directories and filenames customization
#############################################

# root directory for the different benchmark suites
ROOT_BENCH_DIR=./benchmarks

# full path for the target benchmark suite's kernel
BENCH_DIR=$(ROOT_BENCH_DIR)/$(BENCH_NAME)

# parent directory for storing binaries
BIN_DIR=./bin

# executable filenames
CPU_SEQ_BIN=$(BIN_DIR)/$(BENCH_NAME)/cpu_$(SIZE)
OMP_CPU_BIN=$(BIN_DIR)/$(BENCH_NAME)/omp_cpu_$(SIZE)
OMP_GPU_BIN=$(BIN_DIR)/$(BENCH_NAME)/omp_gpu_$(SIZE)

# parent directory for storing executables output
LOGS_DIR=./logs

#############################################
# Kernel custom options
#############################################

# Execute the Makefile for the selected kernel
# Specifies the source files for compilation as well as additional flags (e.g. math libraries)
include $(BENCH_DIR)/src/Makefile

#############################################
# Compiler Options
#############################################

# compiler
CC=gcc

# common flags for the three targets: sequential CPU, parallel CPU, parallel GPU
COMMON_FLAGS=-I $(ROOT_BENCH_DIR)/common

# specific compiler flags for sequential CPU
CPU_SEQ_FLAGS=
# specific compiler flags for parallel CPU target
OMP_CPU_FLAGS=-fopenmp -foffload=disable
# specific compiler flags for parallel GPU target
OMP_GPU_FLAGS=-fopenmp -foffload=nvptx-none=-misa=sm_35


#############################################
# Default target
#############################################

all: compile-cpu compile-omp-cpu compile-omp-gpu

#############################################
# Directory setup targets
#############################################

# creates the directory for binaries
mkdir-bin:
	mkdir -p $(BIN_DIR)/$(BENCH_NAME)

# creates the directory for logs
mkdir-logs:
	mkdir -p $(LOGS_DIR)/$(BENCH_NAME)

#############################################
# Compilation targets
#############################################

# compiles the sequential CPU version
compile-cpu: mkdir-bin
	$(CC) $(COMMON_FLAGS) $(CPU_SEQ_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_CPU_SEQ -DSIZE=$(SIZE) -o $(CPU_SEQ_BIN)

# compiles the parallel GPU version
compile-omp-gpu: mkdir-bin
	$(CC) $(COMMON_FLAGS) $(OMP_CPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_GPU -DSIZE=$(SIZE) -o $(OMP_CPU_BIN)

# compiles the parallel CPU version
compile-omp-cpu: mkdir-bin
	$(CC) $(COMMON_FLAGS) $(OMP_GPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_CPU -DSIZE=$(SIZE) -o $(OMP_GPU_BIN)

#############################################
# Run targets
#############################################

run-cpu: mkdir-logs compile-cpu
	$(CPU_SEQ_BIN)

run-omp-gpu: mkdir-logs compile-omp-gpu
	$(OMP_CPU_BIN)

run-omp-cpu: mkdir-logs compile-omp-cpu
	$(OMP_GPU_BIN)
