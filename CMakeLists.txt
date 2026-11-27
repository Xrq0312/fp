cmake_minimum_required(VERSION 3.20)
project(pointcloud_corner_compare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL CONFIG REQUIRED COMPONENTS common io filters kdtree features segmentation)
find_package(CGAL CONFIG REQUIRED)
find_package(OpenMP)

add_executable(pointcloud_corner_compare
    src/main.cpp
)

add_executable(edge_aware_plane_extract
    src/edge_aware_plane_extract.cpp
)

target_include_directories(pointcloud_corner_compare
    PRIVATE
        ${PCL_INCLUDE_DIRS}
)

target_include_directories(edge_aware_plane_extract
    PRIVATE
        ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(pointcloud_corner_compare
    PRIVATE
        NOMINMAX
        ${PCL_DEFINITIONS}
)

target_compile_definitions(edge_aware_plane_extract
    PRIVATE
        NOMINMAX
        ${PCL_DEFINITIONS}
)

target_link_libraries(pointcloud_corner_compare
    PRIVATE
        ${PCL_LIBRARIES}
)

target_link_libraries(edge_aware_plane_extract
    PRIVATE
        ${PCL_LIBRARIES}
        CGAL::CGAL
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pointcloud_corner_compare PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(edge_aware_plane_extract PRIVATE OpenMP::OpenMP_CXX)
endif()
