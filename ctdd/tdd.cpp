#include "tdd.h"

using namespace std;
using namespace wnode;
using namespace tdd;

void tdd::reset() {
	node::Node<wcomplex>::reset();
	cache::Global_Cache<wcomplex>::reset();
	cache::Global_Cache<CUDAcpl::Tensor>::reset();
}

