cmake_minimum_required(VERSION 3.10)

# 1. Project Setup
project(talawa-ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++-13)
set(CMAKE_CXX_EXTENSIONS OFF) # Forces -std=c++23 instead of -std=gnu++23

# 2. Source Code Auto-Discovery
# 'GLOB_RECURSE' searches for any .cpp file inside src/
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")

# Check for OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP Found! Enabling Multi-threading.")
    link_libraries(OpenMP::OpenMP_CXX)
    add_compile_options(${OpenMP_CXX_FLAGS})
else()
    message(WARNING "OpenMP not found. Matrix operations will be single-threaded.")
endif()

# 3. Create the Library (Only if sources exist)
if(LIB_SOURCES)
    add_library(talawa-ai ${LIB_SOURCES})

    # Define where the headers are
    target_include_directories(talawa-ai PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
    )

    # Compiler warnings
    # --- NEW: Enable AVX2 and FMA optimizations ---
    if(MSVC)
        # Windows (Visual Studio) flags
        # /arch:AVX2 enables AVX2. FMA is usually implied or automatic in modern MSVC.
        # /fp:fast allows aggressive floating point optimizations (reordering math).
        target_compile_options(talawa-ai PRIVATE /W4 /arch:AVX2 /fp:fast)
    else()
        # Linux/macOS (GCC/Clang) flags
        # -mavx2: Enables AVX2 instructions
        # -mfma: Enables Fused Multiply-Add
        # -O3: Max optimization level (required for loop vectorization)
        # target_compile_options(talawa-ai PRIVATE -Wall -Wextra -pedantic -O3 -mavx2 -mfma)

        # -march=native: Optimizes for YOUR CPU architecture (e.g. enables AVX-512)
        # -flto: Link Time Optimization (inlines functions across object files)
        # -funroll-loops: Aggressively unrolls loops
        target_compile_options(talawa-ai PRIVATE
            -Wall -Wextra -pedantic -O3 -march=native -mfma -flto -funroll-loops
        )

        # You also need to link with LTO
        target_link_options(talawa-ai PRIVATE -flto)
    endif()

    message(STATUS "Library 'talawa-ai' configured with ${LIB_SOURCES}")
else()
    message(WARNING "No source files found in src/. Library 'talawa-ai' will NOT be built.")
endif()

# ==========================================
# 4. Auto-Discover & Build Tests
# ==========================================
# This looks into tests/*.test.cpp. For EACH file found, it creates a separate executable.

file(GLOB TEST_SOURCES "tests/*.test.cpp")

if(TEST_SOURCES AND TARGET talawa-ai)
    foreach(test_source ${TEST_SOURCES})
        # Get the filename without extension (e.g., "matrix.test.cpp" -> "matrix.test")
        get_filename_component(test_name ${test_source} NAME_WE)

        # Create an executable for this specific test file
        add_executable(${test_name}.test ${test_source})

        # Link it to your library
        target_link_libraries(${test_name}.test PRIVATE talawa-ai)

        message(STATUS "Added Test: ${test_name}.test")
    endforeach()
endif()

# ==========================================
# 5. Auto-Discover & Build Examples
# ==========================================
# This looks into examples/*.cpp. For EACH file, it creates an executable.

file(GLOB EXAMPLE_SOURCES "examples/*.example.cpp")

if(EXAMPLE_SOURCES AND TARGET talawa-ai)
    foreach(app_source ${EXAMPLE_SOURCES})
        get_filename_component(app_name ${app_source} NAME_WE)

        add_executable(${app_name}.example ${app_source})
        target_link_libraries(${app_name}.example PRIVATE talawa-ai)

        message(STATUS "Added Example: ${app_name}.example")
    endforeach()
endif()

# ==========================================
# 6. Auto-Discover & Build Benchmarks
# ==========================================
# This looks into benchmarks/*.benchmark.cpp. For EACH file, it creates an executable.
file(GLOB BENCHMARK_SOURCES "benchmarks/*.benchmark.cpp")

if(BENCHMARK_SOURCES AND TARGET talawa-ai)
    foreach(bench_source ${BENCHMARK_SOURCES})
        get_filename_component(bench_name ${bench_source} NAME_WE)

        add_executable(${bench_name}.benchmark ${bench_source})
        target_link_libraries(${bench_name}.benchmark PRIVATE talawa-ai)

        message(STATUS "Added Benchmark: ${bench_name}.benchmark")
    endforeach()
endif()
