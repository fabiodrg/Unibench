# README
# 
# Environment variables
# BENCH_NAME: The benchmark suite name and the kernel name. Should match a directory in ./benchmarks. E.g. Polybench/2MM
# SIZE: The problem dimensions (e.g. 512, 1024, 2048, 4096)
# RUNS: The number of consecutive times the kernel should run

check-def-var = $(if $(strip $($1)),,$(error "$1" is not defined))
$(call check-def-var,BENCH_NAME)
$(call check-def-var,SIZE)
$(call check-def-var,RUNS)

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

# logs filenames
CPU_SEQ_LOG=$(LOGS_DIR)/$(BENCH_NAME)/cpu_$(SIZE).log
OMP_CPU_LOG=$(LOGS_DIR)/$(BENCH_NAME)/omp_cpu_$(SIZE).log
OMP_GPU_LOG=$(LOGS_DIR)/$(BENCH_NAME)/omp_gpu_$(SIZE).log

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
	@echo "[INFO] Creating directory $(BIN_DIR)/$(BENCH_NAME)"
	mkdir -p $(BIN_DIR)/$(BENCH_NAME)

# creates the directory for logs
mkdir-logs:
	@echo "[INFO] Creating directory $(LOGS_DIR)/$(BENCH_NAME)"
	mkdir -p $(LOGS_DIR)/$(BENCH_NAME)

#############################################
# Compilation targets
#############################################

# compiles the sequential CPU version
compile-cpu: mkdir-bin
	@echo "[INFO] Compiling $(BENCH_NAME) [CPU, SIZE=$(SIZE)]"
	$(CC) $(COMMON_FLAGS) $(CPU_SEQ_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_CPU_SEQ -DSIZE=$(SIZE) -DN_RUNS=$(RUNS) -o $(CPU_SEQ_BIN)

# compiles the parallel GPU version
compile-omp-cpu: mkdir-bin
	@echo "[INFO] Compiling $(BENCH_NAME) [OMP GPU, SIZE=$(SIZE)]"
	$(CC) $(COMMON_FLAGS) $(OMP_CPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_GPU -DSIZE=$(SIZE) -DN_RUNS=$(RUNS) -o $(OMP_CPU_BIN)

# compiles the parallel CPU version
compile-omp-gpu: mkdir-bin
	@echo "[INFO] Compiling $(BENCH_NAME) [OMP CPU, SIZE=$(SIZE)]"
	$(CC) $(COMMON_FLAGS) $(OMP_GPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_CPU -DSIZE=$(SIZE) -DN_RUNS=$(RUNS) -o $(OMP_GPU_BIN)

#############################################
# Run targets
#############################################

run-cpu: mkdir-logs compile-cpu
	@echo "[INFO] Running $(BENCH_NAME) [CPU, SIZE=$(SIZE)]"
	$(CPU_SEQ_BIN) > $(CPU_SEQ_LOG)

run-omp-gpu: mkdir-logs compile-omp-gpu
	@echo "[INFO] Running $(BENCH_NAME) [OMP GPU, SIZE=$(SIZE)]"
	$(OMP_CPU_BIN) > $(OMP_CPU_LOG)

run-omp-cpu: mkdir-logs compile-omp-cpu
	@echo "[INFO] Running $(BENCH_NAME) [OMP CPU, SIZE=$(SIZE)]"
	$(OMP_GPU_BIN) > $(OMP_GPU_LOG)
