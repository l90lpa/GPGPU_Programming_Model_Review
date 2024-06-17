cmake_minimum_required(VERSION 3.20)

# unset(fmt_DIR CACHE)

project(MatVec_RAJA CXX)

include(CARE/blt/SetupBLT.cmake)

set(CMAKE_CXX_STANDARD 17)
set(APP_DEPENDENCIES care)

if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(APP_DEPENDENCIES ${APP_DEPENDENCIES} cuda)
endif()

add_subdirectory(CARE)

blt_add_executable(
  NAME matvec_raja
  SOURCES matvec_raja.cpp
  DEPENDS_ON ${APP_DEPENDENCIES})

