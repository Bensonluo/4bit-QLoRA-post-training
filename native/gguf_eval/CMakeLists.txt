cmake_minimum_required(VERSION 3.20)
project(tunesmith_gguf_eval LANGUAGES C CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "Existing pinned official llama.cpp checkout")
set(TUNESMITH_LLAMA_REVISION "436f6f89e1e581249900b37a5b8a12a36a6d0912")
if(NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h" OR
   NOT EXISTS "${LLAMA_CPP_DIR}/vendor/nlohmann/json.hpp")
    message(FATAL_ERROR "LLAMA_CPP_DIR must contain the prepared llama.cpp checkout and vendored JSON header")
endif()
find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLAMA_REVISION_STATUS
)
if(NOT LLAMA_REVISION_STATUS EQUAL 0 OR NOT LLAMA_REVISION STREQUAL TUNESMITH_LLAMA_REVISION)
    message(FATAL_ERROR "The native bridge requires llama.cpp ${TUNESMITH_LLAMA_REVISION}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" status --porcelain=v1 --untracked-files=all
    OUTPUT_VARIABLE LLAMA_DIRTY RESULT_VARIABLE LLAMA_STATUS
)
if(NOT LLAMA_STATUS EQUAL 0 OR NOT LLAMA_DIRTY STREQUAL "")
    message(FATAL_ERROR "The native bridge requires an unchanged official llama.cpp checkout")
endif()

# This executable has no serving or network dependency. Keep every backend on CPU
# even when a reused CMake cache contains GPU options from another build.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
foreach(option TESTS TOOLS EXAMPLES SERVER APP MTMD)
    set(LLAMA_BUILD_${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_CPU ON CACHE BOOL "" FORCE)
foreach(option BACKEND_DL METAL CUDA HIP MUSA VULKAN SYCL OPENCL CANN RPC WEBGPU
               ZDNN VIRTGPU VIRTGPU_BACKEND OPENVINO ET ET_SYSEMU HEXAGON ZENDNN OPENMP_FETCH)
    set(GGML_${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama-cpp" EXCLUDE_FROM_ALL)

add_executable(tunesmith-gguf-eval main.cpp)
target_compile_features(tunesmith-gguf-eval PRIVATE cxx_std_17)
target_link_libraries(tunesmith-gguf-eval PRIVATE llama vendor::nlohmann)
target_compile_definitions(tunesmith-gguf-eval PRIVATE
    TUNESMITH_GGUF_REVISION="${TUNESMITH_LLAMA_REVISION}"
)
set_target_properties(tunesmith-gguf-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
)
if(MSVC)
    target_compile_options(tunesmith-gguf-eval PRIVATE /W4)
else()
    target_compile_options(tunesmith-gguf-eval PRIVATE -Wall -Wextra -Wpedantic)
endif()
