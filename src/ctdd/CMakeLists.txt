cmake_minimum_required (VERSION 3.20)


# check whether TDDPY_CONDA_PREFIX is set in CMakePresets.json
if (NOT $ENV{TDDPY_CONDA_PREFIX} STREQUAL "")
    message(STATUS "CURRENT CONDA PREFIX: $ENV{TDDPY_CONDA_PREFIX}")
else()
    message(FATAL_ERROR "The environement variable TDDPY_CONDA_PREFIX is not given. See src/CMakePresets.json")
endif ()


# find the dependencies
list(APPEND CMAKE_PREFIX_PATH $ENV{TDDPY_CONDA_PREFIX})
list(APPEND CMAKE_PREFIX_PATH $ENV{TDDPY_CONDA_PREFIX}/Library)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)

find_package(Boost 1.67.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find the Boost library.")
endif()




# the executable version
add_executable (ctdd_ctest "main.cpp" 
  "ThreadPool.h"
  "typing.hpp"
  "weight.hpp" "cache.hpp"
  "node.hpp" "wnode.hpp" "tdd.hpp")



# the dynamic library version

pybind11_add_module(ctdd_dll "ctddmodule.cpp")
set_target_properties(ctdd_dll 
  PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../tddpy
    SUFFIX $ENV{PYTHON_SHARED_LIBRARY_SUFFIX})

target_link_libraries(ctdd_dll PUBLIC pybind11::module xtensor-python Python::NumPy)
target_compile_definitions(ctdd_dll PRIVATE VERSION_INFO=0.1.0)
