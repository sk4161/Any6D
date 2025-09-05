cmake_minimum_required(VERSION 3.15)
project(mycpp)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -g3 -O3")

# Only Boost headers are needed (algorithm/format/assign). No libs required.
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${BLAS_INCLUDE_DIR}
)

file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

set(PYBIND11_CPP_STANDARD -std=c++17)

pybind11_add_module(mycpp src/app/pybind_api.cpp ${MY_SRC})

# Boost headers only; remove Boost libs from link to avoid missing components
target_link_libraries(mycpp PRIVATE ${OpenMP_CXX_FLAGS} Eigen3::Eigen)
