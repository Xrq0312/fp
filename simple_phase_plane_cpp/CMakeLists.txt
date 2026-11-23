cmake_minimum_required(VERSION 3.20)
project(simple_phase_plane_unwrap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(simple_phase_plane_unwrap simple_phase_plane_unwrap.cpp)
target_link_libraries(simple_phase_plane_unwrap PRIVATE ${OpenCV_LIBS})
target_compile_definitions(simple_phase_plane_unwrap PRIVATE NOMINMAX)
if(MSVC)
    target_compile_options(simple_phase_plane_unwrap PRIVATE /utf-8)
endif()
