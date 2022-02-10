#include "tdd.h"

using namespace wnode;
using namespace tdd;


void TDD::get_inner_data_shape(int64_t* p_storage) const {
	for (int i = 0; i < m_dim_data; i++) {
		p_storage[i] = mp_data_shape[mp_index_order[i]];
	}
	p_storage[m_dim_data] = 2;
}




TDD::~TDD() {
	free(mp_index_order);
	free(mp_parallel_shape);
	//mp_data_shape should not be freed because it points into mp_parallel_shape.
}

weightednode TDD::wnode() const {
	return m_wnode;
}

int TDD::dim_data() const {
	return m_dim_data;
}

const int64_t* TDD::data_shape() const {
	return mp_data_shape;
}

const int* TDD::index_order() const {
	return mp_index_order;
}

int TDD::get_size() const {
	if (m_wnode.p_node == node::TERMINAL_NODE) {
		return 0;
	}
	return m_wnode.p_node->get_size();
}

TDD* TDD::as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const int* p_index_order) {
	auto dim_total = t.dim()-1;
	auto dim_data = dim_total - dim_parallel;
	//use one int* to store parallel shape and data shape together.
	int64_t* p_parallel_shape = (int64_t*)malloc(sizeof(int64_t) * (dim_total + 1));
	//this is the extra inner dimension for CUDAcpl
	p_parallel_shape[dim_total] = 2;
	for (int i = 0; i < dim_total; i++) {
		p_parallel_shape[i] = t.size(i);
	}

	//prepare data_shape
	int64_t* p_data_shape = p_parallel_shape + dim_parallel;
	//prepare index_order
	int* p_index_order_pd = (int*)malloc(sizeof(int) * dim_data);
	if (p_index_order == nullptr) {
		for (int i = 0; i < dim_data; i++) {
			p_index_order_pd[i] = i;
		}
	}
	else {
		for (int i = 0; i < dim_data; i++) {
			p_index_order_pd[i] = p_index_order[i];
		}
	}

	weightednode w_node = as_tensor_iterate(t, dim_parallel, p_parallel_shape, dim_data, p_data_shape, p_index_order_pd, 0);

	TDD* p_res = new TDD();
	//note the ownership transfer here
	p_res->m_wnode = w_node;
	p_res->m_dim_parallel = dim_parallel;
	p_res->m_dim_data = dim_data;
	p_res->mp_index_order = p_index_order_pd;
	p_res->mp_parallel_shape = p_parallel_shape;
	p_res->mp_data_shape = p_data_shape;

	return p_res;
}

CUDAcpl::Tensor TDD::CUDAcpl() const {
	int64_t* p_inner_shape = (int64_t*)malloc(sizeof(int64_t) * (m_dim_data + 1));
	get_inner_data_shape(p_inner_shape);
	CUDAcpl::Tensor res = wnode::to_CUDAcpl(m_wnode, m_dim_data, p_inner_shape);
	free(p_inner_shape);
	return res;
}
