# Doc for TddPy

## Environment  Settings

C++ dependencies: Boost, LibTorch, Python 3.9, Visual Studio

Configurations should be reset first, especially the include path and .lib files.

1. An extra environment variable 'Boost' should be set to the C++ Boost library location.
2. An extra environment variable 'libtorch' should be set to the C++ LibTorch library location.

Note that some adjustments of the project properties in Visual Studio may be needed.

## Project Structure

- ctdd: the C++ backend for TddPy
  - stdafx.h
  - cache.hpp: the module for all kinds of unique tables
  - config.h: constants used in this tool
  - ctdd.cpp, ctdd.h: wrapper of tdd objects for the C/Python interface
  - ctddmodule.cpp: the C/Python interface (build configuration only)
  - CUDAcpl.cpp, CUDAcpl.h: the warpping as complex numbers for libtorch tensors
  - main_test.cpp: the main() entrance for testing (Inner configuration only)
  - manage.cpp, manage.hpp: the resource management module, including memory monitor and thread control
  - node.hpp: the code for nodes in the TDD
  - simpletools.h: simple methods to deal with arrays
  - tdd.cpp, tdd.hpp: the code for the TDD data structure
  - ThreadPool.h: a thread pool module from the popular GitHub project (https://github.com/progschj/ThreadPool)
  - weight.hpp: the code for dealing with weights in the TDD
  - wnode.hpp: the data structure of "weighted node". It turns out that this is the appropriate building block of TDD.
- tddpy: the Python wrapper of the C++ backend
  - node.py: the interfaces of nodes in the TDD
  - tdd.py: the interfaces of the TDD data structure
  - global_method.py: the interfaces for global methods, including cache clearing, thread number settings and so on
  - abstract_coordinator: the abstract class for order coordinators
  - trival_coordinator: the implementation of a trival coordinator (consistent with the convention of tensordot in numpy and pytorch)
  - global_order_coordinator: the implmenetation of a global order coordinator (consistent with the convention in Xin Hong's paper)