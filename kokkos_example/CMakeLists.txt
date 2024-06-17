cmake_minimum_required(VERSION 3.16)

project(MatVec_Kokkos CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(kokkos)

add_executable(matvec_kokkos matvec_kokkos.cpp)

target_link_libraries(matvec_kokkos Kokkos::kokkos)

