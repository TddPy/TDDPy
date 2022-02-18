#pragma once

#ifdef NDEBUG
#include <Python.h>
#include <torch/python.h>
#endif

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

#include <torch/script.h>
#include <torch/torch.h>


#include "simpletools.h"
#include "config.h"
#include "CUDAcpl.h"