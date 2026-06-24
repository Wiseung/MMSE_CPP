cmake_minimum_required(VERSION 3.31)

project(MMSE_CPP LANGUAGES CXX)

option(MMSE_BUILD_TESTS "Build MMSE unit tests" ON)
option(MMSE_BUILD_BENCH "Build MMSE benchmark" ON)

add_library(mmse_cpp
    src/mmse_equalizer_cpu.cpp
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
    target_compile_options(mmse_cpp PRIVATE /W4 /permissive- /EHsc)
    set_source_files_properties(
        src/mmse_avx2.cpp
        PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2"
    )
else()
    target_compile_options(mmse_cpp PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(
        src/mmse_avx2.cpp
        PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma"
    )
endif()

target_compile_definitions(mmse_cpp PRIVATE NOMINMAX)

if(MMSE_BUILD_TESTS)
    enable_testing()

    add_executable(mmse_tests tests/mmse_tests.cpp)
    target_link_libraries(mmse_tests PRIVATE mmse_cpp)
    target_include_directories(mmse_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME mmse_tests COMMAND mmse_tests)
endif()

if(MMSE_BUILD_BENCH)
    add_executable(mmse_bench bench/mmse_bench.cpp)
    target_link_libraries(mmse_bench PRIVATE mmse_cpp)
    target_include_directories(mmse_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
