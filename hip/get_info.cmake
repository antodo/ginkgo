set(log_types "detailed_log")
foreach(log_type ${log_types})
    ginkgo_print_module_header(${${log_type}} "HIP")
    set(print_var "GINKGO_HIPCONFIG_PATH;GINKGO_HIP_AMDGPU;GINKGO_HIP_HCC_COMPILER_FLAGS;GINKGO_HIP_NVCC_COMPILER_FLAGS;GINKGO_HIP_THRUST_PATH"
        )
    foreach(var ${print_var})
        ginkgo_print_variable(${${log_type}} ${var} )
    endforeach()
endforeach()
foreach(log_type ${log_types})
    ginkgo_print_module_footer(${${log_type}} "HIP variables:")
    set(print_var "HIP_VERSION;HIP_COMPILER;HIP_PATH;ROCM_PATH;HIP_PLATFORM;HIP_ROOT_DIR;HCC_PATH;HIP_RUNTIME;HIPBLAS_PATH;HIPSPARSE_PATH;HIP_CLANG_INCLUDE_PATH;HIP_CLANG_PATH;HIP_HIPCC_EXECUTABLE;HIP_HIPCONFIG_EXECUTABLE;HIP_HOST_COMPILATION_CPP"
        )
    foreach(var ${print_var})
        ginkgo_print_variable(${${log_type}} ${var} )
    endforeach()
    ginkgo_print_flags(${detailed_log} "HIP_HCC_FLAGS")
    ginkgo_print_flags(${detailed_log} "HIP_HIPCC_FLAGS")
    ginkgo_print_flags(${detailed_log} "HIP_NVCC_FLAGS")
    ginkgo_print_module_footer(${detailed_log} "")
endforeach()
