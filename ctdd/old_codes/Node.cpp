#include "Node.h"
#include "weights.h"
#include "CUDAcpl.h"

using namespace cache;
using namespace node;


/*
	Implementations of Node.
*/

int node::Node<wcomplex>::m_global_id = 0;
int node::Node<CUDAcpl::Tensor>::m_global_id = 0;
unique_table<wcomplex> node::Node<wcomplex>::m_unique_table = unique_table<wcomplex>();
unique_table<CUDAcpl::Tensor> node::Node<CUDAcpl::Tensor>::m_unique_table = unique_table<CUDAcpl::Tensor>();
