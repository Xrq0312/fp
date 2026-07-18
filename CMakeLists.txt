cmake_minimum_required(VERSION 3.20)
project(edge_aware_plane_extract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL CONFIG REQUIRED COMPONENTS common io filters kdtree features segmentation)
find_package(OpenMP)

add_executable(edge_aware_plane_extract
    src/edge_aware_plane_extract.cpp
)

target_include_directories(edge_aware_plane_extract
    PRIVATE
        ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(edge_aware_plane_extract
    PRIVATE
        NOMINMAX
        ${PCL_DEFINITIONS}
)

target_link_libraries(edge_aware_plane_extract
    PRIVATE
        ${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(edge_aware_plane_extract PRIVATE OpenMP::OpenMP_CXX)
endif()
