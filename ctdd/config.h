#pragma once

#include <chrono>

typedef std::complex<double> wcomplex;

const double DEFAULT_EPS = 3E-7;

const int DEFAULT_THREAD_NUM = 4;

const uint64_t DEFAULT_VMEM_LIMIT = 5000. * 1024. * 1024.;

const std::chrono::duration<double> DEFAULT_MEM_CHECK_PERIOD = std::chrono::duration<double>(0.5);


//#define DECONSTRUCTOR_DEBUG