#pragma once

typedef std::complex<double> wcomplex;

const double DEFAULT_EPS = 3E-7;

const int DEFAULT_THREAD_NUM = 4;

const int DEFAULT_CLEANER_THREAD_NUM = 1;

const uint64_t DEFAULT_VMEM_LIMIT = 8 * 1024. * 1024. * 1024.;

const double DEFAULT_MEM_CHECK_PERIOD = 0.5;

// the info line num in /proc/{pid}/status file
#define VMRSS_LINE 22

//#define VMEM_SHUT_DOWN
//#define RESOURCE_OUTPUT

//#define NO_LOCK_TEST