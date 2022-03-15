#pragma once

#include <Python.h>
#include <torch/python.h>


#include <cmath>
#include <iostream>
#include <complex>
#include <boost/unordered_map.hpp>
#include <string>
#include <boost/container_hash/hash_fwd.hpp>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <vector>
#include <assert.h>
#include <shared_mutex>
#include <chrono>

#include <torch/script.h>
#include <torch/torch.h>

// resource management

#include <windows.h>  
#include <psapi.h>  
//#include <tlhelp32.h>
#include <direct.h>
#include <process.h>



#include "simpletools.h"
#include "config.h"
#include "CUDAcpl.h"
#include "ThreadPool.h"