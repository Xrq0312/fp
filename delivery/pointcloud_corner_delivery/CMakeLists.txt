cmake_minimum_required(VERSION 3.20)
project(pointcloud_corner_compare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL CONFIG REQUIRED COMPONENTS common io filters kdtree)
find_package(OpenMP)

add_executable(pointcloud_corner_compare
    src/main.cpp
)

target_include_directories(pointcloud_corner_compare
    PRIVATE
        ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(pointcloud_corner_compare
    PRIVATE
        NOMINMAX
        ${PCL_DEFINITIONS}
)

target_link_libraries(pointcloud_corner_compare
    PRIVATE
        ${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pointcloud_corner_compare PRIVATE OpenMP::OpenMP_CXX)
endif()
