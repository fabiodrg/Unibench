# README
# 
# Environment variables
# BENCH_NAME: The benchmark suite name and the kernel name. Should match a directory in ./benchmarks. E.g. Polybench/2MM
# SIZE: The problem dimensions (e.g. 512, 1024, 2048, 4096)
# RUNS: The number of consecutive times the kernel should run

#check-def-var = $(if $(strip $($1)),,$(error "$1" is not defined))
#$(call check-def-var,BENCH_NAME)
#$(call check-def-var,SIZE)
#$(call check-def-var,RUNS)

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
LOGS_DIR=./logs-laptop-O0

# logs filenames
CPU_SEQ_LOG=$(LOGS_DIR)/$(BENCH_NAME)/cpu_$(SIZE).log
OMP_CPU_LOG=$(LOGS_DIR)/$(BENCH_NAME)/omp_cpu_$(SIZE).log
OMP_GPU_LOG=$(LOGS_DIR)/$(BENCH_NAME)/omp_gpu_$(SIZE).log
LLVM_MCA_LOG=$(LOGS_DIR)/$(BENCH_NAME)/llvm_mca.log

#############################################
# Compiler Options
#############################################

# compiler
CC=gcc
# includes
INCLUDE=-I $(ROOT_BENCH_DIR)/common
# compiler flags
CFLAGS=-O0
# libraries
LDLIBS=
# specific compiler flags for sequential CPU
TARGET_CPU_FLAGS=
# specific compiler flags for parallel CPU target
TARGET_OMP_CPU_FLAGS=-fopenmp -foffload=disable
# specific compiler flags for parallel GPU target
TARGET_OMP_GPU_FLAGS=-fopenmp -foffload=nvptx-none=-misa=sm_35

#############################################
# Kernel custom options
#############################################

# Execute the Makefile for the selected kernel
# Specifies the source files for compilation as well as additional flags (e.g. math libraries)
include $(BENCH_DIR)/src/Makefile

# common compiling command
CC_COMMON=$(CC) $(CFLAGS) $(INCLUDE) $(LDLIBS)

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
	$(CC_COMMON) $(TARGET_CPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_CPU_SEQ -DSIZE=$(SIZE) -DN_RUNS=$(RUNS) -o $(CPU_SEQ_BIN)

# compiles the parallel GPU version
compile-omp-gpu: mkdir-bin
	@echo "[INFO] Compiling $(BENCH_NAME) [OMP GPU, SIZE=$(SIZE)]"
	$(CC_COMMON) $(TARGET_OMP_GPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_GPU -DSIZE=$(SIZE) -DN_RUNS=$(RUNS) -o $(OMP_GPU_BIN)

# compiles the parallel CPU version
compile-omp-cpu: mkdir-bin
	@echo "[INFO] Compiling $(BENCH_NAME) [OMP CPU, SIZE=$(SIZE)]"
	$(CC_COMMON) $(TARGET_OMP_CPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_CPU -DSIZE=$(SIZE) -DN_RUNS=$(RUNS) -o $(OMP_CPU_BIN)

#############################################
# Run targets
#############################################

run-cpu: mkdir-logs compile-cpu
	@echo "[INFO] Running $(BENCH_NAME) [CPU, SIZE=$(SIZE)]"
	stdbuf -oL $(CPU_SEQ_BIN) > $(CPU_SEQ_LOG)

run-omp-gpu: mkdir-logs compile-omp-gpu
	@echo "[INFO] Running $(BENCH_NAME) [OMP GPU, SIZE=$(SIZE)]"
	stdbuf -oL $(OMP_GPU_BIN) > $(OMP_GPU_LOG)

run-omp-cpu: mkdir-logs compile-omp-cpu
	@echo "[INFO] Running $(BENCH_NAME) [OMP CPU, SIZE=$(SIZE)]"
	stdbuf -oL $(OMP_CPU_BIN) > $(OMP_CPU_LOG)

#############################################
# Run test mode
#############################################
test-cpu:
	@echo "[INFO] Compiling $(BENCH_NAME) [OMP CPU, Test mode]"
	$(CC_COMMON) $(TARGET_OMP_CPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_CPU -DRUN_TEST -o test_cpu
	@echo "[INFO] Launching..."
	./test_cpu
	@echo "[INFO] Completed"

test-gpu:
	@echo "[INFO] Compiling $(BENCH_NAME) [OMP GPU, Test mode]"
	$(CC_COMMON) $(TARGET_OMP_GPU_FLAGS) $(BENCH_FLAGS) $(SRC_OBJS) -DRUN_OMP_GPU -DRUN_TEST -o test_gpu
	@echo "[INFO] Launching..."
	./test_gpu
	@echo "[INFO] Completed"

test: test-cpu test-gpu


#############################################
# Run LLVM MCA
#############################################
llvm-mca:
	$(CC_COMMON) -S $(SRC_OBJS) -DRUN_CPU_SEQ -DLLVM_MCA\
		-o /dev/stdout | llvm-mca --iterations=100 > $(LLVM_MCA_LOG)

#############################################
# Debug (for profiling)
#############################################

__debug:
	$(eval RUNS=1)
	$(eval CC_COMMON+=-g)

debug: __debug compile-cpu
	
