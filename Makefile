# README
# 
# Environment variables
# BENCH_NAME: The benchmark suite name and the kernel name. Should match a directory in ./benchmarks. E.g. Polybench/2MM
# SIZE: The problem dimensions (e.g. MINI, SMALL, MEDIUM, LARGE)
# RUNS: The number of consecutive times the kernel should run

ifndef IN_RUNS
IN_RUNS=1
endif

# Include Makefile with user settings
include ./Makefile.defs

# Include Makefile with auxiliars
include ./Utils.mk

# full path for the target benchmark suite's kernel
BENCH_DIR=$(ROOT_BENCH_DIR)/$(BENCH_NAME)

# Include the Makefile for the selected kernel
# Specifies the source files for compilation, libraries to be linked, etc.
include $(BENCH_DIR)/src/Makefile

# common compiling command
CC_COMMON=$(CC) $(CFLAGS) $(C_INCLUDE_PATH) $(LDLIBS) $(LDFLAGS)

# executable filenames
CPU_SEQ_BIN=$(BIN_DIR)/cpu_$(SIZE)
OMP_CPU_BIN=$(BIN_DIR)/omp_cpu_$(SIZE)
OMP_GPU_BIN=$(BIN_DIR)/omp_gpu_$(SIZE)

# logs filenames
CPU_SEQ_LOG=$(LOGS_DIR)/cpu_$(SIZE).log
OMP_CPU_LOG=$(LOGS_DIR)/omp_cpu_$(SIZE).log
OMP_GPU_LOG=$(LOGS_DIR)/omp_gpu_$(SIZE).log
LLVM_MCA_LOG=$(LOGS_DIR)/llvm_mca.log

#############################################
# Default target
#############################################

all: compile-cpu compile-omp-cpu compile-omp-gpu

#############################################
# Directory setup targets
#############################################

# creates the directory for binaries
mkdir-bin:
	@echo "[INFO] Creating directory $(BIN_DIR)"
	@mkdir -p $(BIN_DIR)

# creates the directory for logs
mkdir-logs:
	@echo "[INFO] Creating directory $(LOGS_DIR)"
	@mkdir -p $(LOGS_DIR)

#############################################
# Compilation targets
#############################################

# Args:
# - Target specific compilation flags
# - The target device ( RUN_CPU_SEQ | RUN_OMP_CPU | RUN_OMP_GPU )
# - Other Bench specific env variables (-DSIZE, -DTEST, ...)
# - The output binary filename
ifeq "$(COMPILER)" "GNU"
# If GNU, pass the "GCC_OFFLOAD_FLAGS"
define compile
	@echo "$(CC_COMMON) $(1) $(GCC_OFFLOAD_FLAGS) -D$(2) $(3) $(SRC_OBJS) -o $(4) $(LDLIBS)"
	$(CC_COMMON) $(1) $(GCC_OFFLOAD_FLAGS) -D$(2) $(3) $(SRC_OBJS) -o $(4) $(LDLIBS)
endef
else
define compile
	@echo "$(CC_COMMON) $(1) -D$(2) $(3) $(SRC_OBJS) -o $(4)"
	$(CC_COMMON) $(1) -D$(2) $(3) $(SRC_OBJS) -o $(4)
endef
endif

# Args:
# - The target device ( RUN_CPU_SEQ | RUN_OMP_CPU | RUN_OMP_GPU )
# - The output binary filename
# - Target specific compilation flags
define device_compile
	$(call compile,$(3),$(1),-D$(SIZE) -DIN_RUNS=$(IN_RUNS),$(2))
endef

# Args:
# - The target device ( RUN_CPU_SEQ | RUN_OMP_CPU | RUN_OMP_GPU )
# - Target specific compilation flags
# - The output binary filename
define device_test
	$(call compile,$(2),$(1),-DRUN_TEST,$(3))
endef

# compiles the sequential CPU version
compile-cpu: mkdir-bin
	@$(call log_info,"Compiling $(BENCH_NAME) [CPU - SIZE=$(SIZE)]")
	@$(call device_compile,RUN_CPU_SEQ,$(CPU_SEQ_BIN))

# compiles the parallel GPU version
compile-omp-gpu: mkdir-bin
	@$(call log_info,"Compiling $(BENCH_NAME) [OMP GPU - SIZE=$(SIZE)]")
	@$(call device_compile,RUN_OMP_GPU,$(OMP_GPU_BIN),$(OMP_OFFLOAD_GPU))

# compiles the parallel CPU version
compile-omp-cpu: mkdir-bin
	@$(call log_info,"Compiling $(BENCH_NAME) [OMP CPU - SIZE=$(SIZE)]")
	@$(call device_compile,RUN_OMP_CPU,$(OMP_CPU_BIN),$(OMP_OFFLOAD_CPU))

#############################################
# Run test mode
#############################################
test-cpu:
	@$(call log_info,"Compiling $(BENCH_NAME) [OMP CPU - Test mode]")
	@$(call device_test,RUN_OMP_CPU,$(OMP_OFFLOAD_CPU),test_cpu)
	@$(call log_info,"Launching...")
	@./test_cpu && $(call log_success,"No errors!") || $(call log_error,"Detected errors!")
	@rm ./test_cpu

test-gpu:
	@$(call log_info,"Compiling $(BENCH_NAME) [OMP GPU - Test mode]")
	@$(call device_test,RUN_OMP_GPU,$(OMP_OFFLOAD_GPU),test_gpu)
	@$(call log_info,"Launching...")
	@./test_gpu && $(call log_success,"No errors!") || $(call log_error,"Detected errors!")
	@rm ./test_gpu

test: test-cpu test-gpu

#############################################
# Run targets
#############################################

# Args:
# - The log filename
# - The binary filename
define run
	@date > $(1)
	@for i in `seq 1 $(RUNS)`; do stdbuf -oL $(2) >> $(1); done
	@$(call log_info,"Log available at: $(1)")
endef

run-cpu: mkdir-logs compile-cpu
	@$(call log_info,"Running $(BENCH_NAME) [CPU - SIZE=$(SIZE)]")
	@$(call run,$(CPU_SEQ_LOG),$(CPU_SEQ_BIN))

run-omp-gpu: mkdir-logs compile-omp-gpu
	@$(call log_info,"Running $(BENCH_NAME) [OMP GPU - SIZE=$(SIZE)]")
	@$(call run,$(OMP_GPU_LOG),$(OMP_GPU_BIN))

run-omp-cpu: mkdir-logs compile-omp-cpu
	@$(call log_info,"Running $(BENCH_NAME) [OMP CPU - SIZE=$(SIZE)]")
	@$(call run,$(OMP_CPU_LOG),$(OMP_CPU_BIN))

#############################################
# Run LLVM MCA
#############################################
llvm-mca:
	$(CC_COMMON) -S $(SRC_OBJS) -DRUN_CPU_SEQ -DLLVM_MCA\
		-o /dev/stdout | llvm-mca --iterations=128 --bottleneck-analysis > $(LLVM_MCA_LOG)

#############################################
# Debug (for profiling)
#############################################

__debug:
	$(eval CC_COMMON+=-g)

debug-cpu: __debug compile-cpu

debug-omp-cpu: __debug compile-omp-cpu

__asm_common:
	$(eval RUNS=1)
	$(eval CC_COMMON+=-S -fverbose-asm -masm=intel)
	$(eval CPU_SEQ_BIN=${CPU_SEQ_BIN}.s)
	$(eval OMP_CPU_BIN=${OMP_CPU_BIN}.s)

asm: __asm_common compile-cpu

asm_omp: __asm_common compile-omp-cpu

#############################################
# Profile
#############################################

gpu-trace: compile-omp-gpu
	@mkdir -p $(PROF_GPU_DIR)
	nvprof --print-gpu-trace --concurrent-kernels off --profile-api-trace none \
	--csv --log-file $(PROF_GPU_DIR)/gpu_trace_$(SIZE).csv \
	$(OMP_GPU_BIN)
