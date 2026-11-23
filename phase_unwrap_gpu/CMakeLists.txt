cmake_minimum_required(VERSION 3.24)
project(phase_unwrap_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs)

add_executable(phase_unwrap_gpu src/main.cu)
target_include_directories(phase_unwrap_gpu PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(phase_unwrap_gpu PRIVATE ${OpenCV_LIBS})
target_compile_definitions(phase_unwrap_gpu PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)

# CUDA 12.6 predates Blackwell cubins. Shipping Ampere/Ada cubins plus PTX lets
# newer drivers JIT the kernel on RTX 50-series cards while retaining portability.
set_target_properties(phase_unwrap_gpu PROPERTIES
  CUDA_ARCHITECTURES "86-real;89-real;89-virtual"
  CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
  target_compile_options(phase_unwrap_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /Ob3 /GL>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --restrict --extra-device-vectorization -Xcompiler=/O2,/Ob3>)
  target_link_options(phase_unwrap_gpu PRIVATE /LTCG)
else()
  target_compile_options(phase_unwrap_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --restrict --extra-device-vectorization>)
endif()

