


# check whether TDDPY_CONDA_PREFIX is set in CMakePresets.json
if (NOT $ENV{TDDPY_CONDA_PREFIX} STREQUAL "")
    message(STATUS "CURRENT CONDA PREFIX: $ENV{TDDPY_CONDA_PREFIX}")
else()
    message(FATAL_ERROR "The environement variable TDDPY_CONDA_PREFIX is not given. See .vscode/settings.json")
endif ()


# the executable version
add_executable (ctdd_ctest "main.cpp" "ctdd.h")
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ctdd_ctest PROPERTY CXX_STANDARD 20)
endif()

# the dynamic library version
add_library(ctdd_dll "ctddmodule.cpp")
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ctdd_dll PROPERTY CXX_STANDARD 20)
endif()


# add the include directories
include_directories($ENV{TDDPY_CONDA_PREFIX}/include)
include_directories($ENV{TDDPY_CONDA_PREFIX}/Library/include)
include_directories($ENV{TDDPY_CONDA_PREFIX}/Lib/site-packages/numpy/core/include)


# link the static libraries
add_library(pythonlib STATIC IMPORTED)
set_target_properties(pythonlib PROPERTIES IMPORTED_LOCATION $ENV{TDDPY_CONDA_PREFIX}/libs/python311.lib)
target_link_libraries(ctdd_dll python)
