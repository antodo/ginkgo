ginkgo_print_module_header(${detailed_log} "CUDA")
ginkgo_print_variable(${detailed_log} "GINKGO_CUDA_ARCHITECTURES")
ginkgo_print_variable(${detailed_log} "GINKGO_CUDA_COMPILER_FLAGS")
ginkgo_print_variable(${detailed_log} "GINKGO_CUDA_DEFAULT_HOST_COMPILER")
ginkgo_print_variable(${detailed_log} "GINKGO_CUDA_ARCH_FLAGS")
ginkgo_print_module_footer(${detailed_log} "CUDA variables:")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_COMPILER")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_COMPILER_VERSION")
ginkgo_print_flags(${detailed_log} "CMAKE_CUDA_FLAGS")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_HOST_COMPILER")
ginkgo_print_variable(${detailed_log} "CUDA_INCLUDE_DIRS")
ginkgo_print_module_footer(${detailed_log} "CUDA Libraries:")
ginkgo_print_variable(${detailed_log} "CUBLAS")
ginkgo_print_variable(${detailed_log} "CUDA_RUNTIME_LIBS")
ginkgo_print_variable(${detailed_log} "CUSPARSE")
ginkgo_print_module_footer(${detailed_log} "")
