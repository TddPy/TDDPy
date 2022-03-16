#pragma once

#include <Python.h>
#include <torch/python.h>


#include <cmath>
#include <iostream>
#include <complex>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/container_hash/hash_fwd.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <vector>
#include <assert.h>
#include <chrono>

#include <torch/script.h>
#include <torch/torch.h>

// resource management

#include <windows.h>  
#include <psapi.h>  
//#include <tlhelp32.h>
#include <direct.h>
#include <process.h>

// multi-thread
#include <shared_mutex>
#include <atomic>
#include "ThreadPool.h"

#include "simpletools.h"
#include "config.h"
#include "CUDAcpl.h"
