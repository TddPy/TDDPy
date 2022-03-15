#include "manage.hpp"

uint64_t mng::vmem_limit = DEFAULT_VMEM_LIMIT;
HANDLE mng::current_process;
std::chrono::duration<double> mng::mem_check_period = DEFAULT_MEM_CHECK_PERIOD;