#include "tdd.h"

using namespace std;
using namespace wnode;
using namespace tdd;

void tdd::reset() {
	node::Node<wcomplex>::reset();
	dict::global_duplicate_cache_w = dict::duplicate_table<wcomplex>();
	dict::global_shift_cache_w = dict::duplicate_table<wcomplex>();
	dict::global_CUDAcpl_cache_w = dict::CUDAcpl_table<wcomplex>();
	dict::global_sum_cache_w = dict::sum_table<wcomplex>();
	dict::global_cont_cache_w = dict::cont_table<wcomplex>();
}

