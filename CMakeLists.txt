cmake_minimum_required(VERSION 3.31)

project(MMSE_CPP LANGUAGES CXX)

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
    src/thread_pool.cpp
)

target_include_directories(mmse_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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
    set_target_properties(mmse_cpp PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
    set_source_files_properties(
        src/mmse_cuda_runtime.cu
        PROPERTIES
        LANGUAGE CUDA
    )
endif()

target_compile_definitions(mmse_cpp PRIVATE NOMINMAX)
target_compile_definitions(mmse_cpp PRIVATE MMSE_CUDA_ENABLED=$<BOOL:${MMSE_CUDA_ENABLED}>)

include(GNUInstallDirs)

install(TARGETS mmse_cpp
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES README.md
    DESTINATION .
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
    install(TARGETS mmse_channel_budget
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
