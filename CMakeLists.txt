cmake_minimum_required(VERSION 3.31)

project(MMSE_CPP LANGUAGES CXX)

include(GNUInstallDirs)

option(MMSE_BUILD_TESTS "Build MMSE unit tests" ON)
option(MMSE_BUILD_BENCH "Build MMSE benchmark" ON)
option(MMSE_ENABLE_CUDA "Enable CUDA backend when toolkit is available" ON)

set(MMSE_CUDA_SOURCES src/mmse_cuda_runtime_stub.cpp)
set(MMSE_CUDA_ENABLED OFF)

if(MMSE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(MMSE_CUDA_SOURCES src/mmse_cuda_runtime.cu)
        set(MMSE_CUDA_ENABLED ON)
    endif()
endif()

add_library(mmse_cpp
    src/mmse_equalizer_cpu.cpp
    src/mmse_equalizer_gpu.cpp
    ${MMSE_CUDA_SOURCES}
    src/mmse_avx2.cpp
    src/pdcch_tail_biting_decoder.cpp
    src/thread_pool.cpp
)
add_library(MMSE_CPP::mmse_cpp ALIAS mmse_cpp)

target_include_directories(mmse_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(mmse_cpp PUBLIC cxx_std_20)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_compile_options(mmse_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc>
    )
    target_link_options(mmse_cpp PRIVATE /NODEFAULTLIB:LIBCMT)
    set_source_files_properties(
        src/mmse_avx2.cpp
        PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2"
    )
else()
    target_compile_options(mmse_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
    set_source_files_properties(
        src/mmse_avx2.cpp
        PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma"
    )
endif()

if(MMSE_CUDA_ENABLED)
    set_target_properties(mmse_cpp PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES 89
    )
    set_source_files_properties(
        src/mmse_cuda_runtime.cu
        PROPERTIES
        LANGUAGE CUDA
    )
endif()

target_compile_definitions(mmse_cpp PRIVATE NOMINMAX)
target_compile_definitions(mmse_cpp PRIVATE MMSE_CUDA_ENABLED=$<BOOL:${MMSE_CUDA_ENABLED}>)

include(CMakePackageConfigHelpers)

set(MMSE_CPP_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/MMSE_CPP")
configure_package_config_file(
    cmake/MMSE_CPPConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MMSE_CPPConfig.cmake"
    INSTALL_DESTINATION "${MMSE_CPP_CONFIG_INSTALL_DIR}"
)

install(TARGETS mmse_cpp
    EXPORT MMSE_CPPTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT MMSE_CPPTargets
    FILE MMSE_CPPTargets.cmake
    NAMESPACE MMSE_CPP::
    DESTINATION "${MMSE_CPP_CONFIG_INSTALL_DIR}"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/MMSE_CPPConfig.cmake"
    DESTINATION "${MMSE_CPP_CONFIG_INSTALL_DIR}"
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES README.md
    DESTINATION .
)

install(FILES docs/pdcch_pdsch_handoff_sdk_v1.md
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/MMSE_CPP
)

if(MMSE_BUILD_TESTS)
    enable_testing()

    add_executable(mmse_tests tests/mmse_tests.cpp)
    target_link_libraries(mmse_tests PRIVATE mmse_cpp)
    target_include_directories(mmse_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(mmse_tests PRIVATE MMSE_CUDA_ENABLED=$<BOOL:${MMSE_CUDA_ENABLED}>)
    if(MSVC)
        target_link_options(mmse_tests PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    add_test(NAME mmse_tests COMMAND mmse_tests)
    install(TARGETS mmse_tests
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(MMSE_BUILD_BENCH)
    add_executable(mmse_bench bench/mmse_bench.cpp)
    target_link_libraries(mmse_bench PRIVATE mmse_cpp)
    target_include_directories(mmse_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    install(TARGETS mmse_bench
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(pdcch_module_demo bench/pdcch_module_demo.cpp)
    target_link_libraries(pdcch_module_demo PRIVATE mmse_cpp)
    target_include_directories(pdcch_module_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    install(TARGETS pdcch_module_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(pdcch_pdsch_handoff_demo bench/pdcch_pdsch_handoff_demo.cpp)
    target_link_libraries(pdcch_pdsch_handoff_demo PRIVATE mmse_cpp)
    install(TARGETS pdcch_pdsch_handoff_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(pdcch_decode_bench bench/pdcch_decode_bench.cpp)
    target_link_libraries(pdcch_decode_bench PRIVATE mmse_cpp)
    target_include_directories(pdcch_decode_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    install(TARGETS pdcch_decode_bench
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(pdcch_gpu_decode_bench bench/pdcch_gpu_decode_bench.cpp)
    target_link_libraries(pdcch_gpu_decode_bench PRIVATE mmse_cpp)
    target_include_directories(pdcch_gpu_decode_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(MSVC)
        target_link_options(pdcch_gpu_decode_bench PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    install(TARGETS pdcch_gpu_decode_bench
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(pdcch_ncu_workload bench/pdcch_ncu_workload.cpp)
    target_link_libraries(pdcch_ncu_workload PRIVATE mmse_cpp)
    target_include_directories(pdcch_ncu_workload PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(MSVC)
        target_link_options(pdcch_ncu_workload PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    install(TARGETS pdcch_ncu_workload
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(mmse_cuda_profile bench/mmse_cuda_profile.cpp)
    target_link_libraries(mmse_cuda_profile PRIVATE mmse_cpp)
    target_include_directories(mmse_cuda_profile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(mmse_cuda_profile PRIVATE MMSE_CUDA_ENABLED=$<BOOL:${MMSE_CUDA_ENABLED}>)
    if(MSVC)
        target_link_libraries(mmse_cuda_profile PRIVATE Psapi)
        target_link_options(mmse_cuda_profile PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    install(TARGETS mmse_cuda_profile
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    add_executable(mmse_channel_budget bench/mmse_channel_budget.cpp)
    target_link_libraries(mmse_channel_budget PRIVATE mmse_cpp)
    target_include_directories(mmse_channel_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(MSVC)
        target_link_options(mmse_channel_budget PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    install(TARGETS mmse_channel_budget
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
