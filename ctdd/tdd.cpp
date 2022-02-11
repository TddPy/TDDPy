#include "tdd.h"

using namespace std;
using namespace wnode;
using namespace tdd;


void TDD::get_inner_data_shape(int64_t* p_storage) const {
	for (int i = 0; i < m_dim_data; i++) {
		p_storage[i] = mp_data_shape[mp_index_order[i]];
	}
	p_storage[m_dim_data] = 2;
}


void TDD::__print() const {
	cout << "weight: " << m_wnode.weight << endl;
	cout << "node: " << m_wnode.p_node << endl;
	cout << "dim_parallel: " << m_dim_parallel << endl;
	cout << "parallel_shape: (";
	for (int i = 0; i < m_dim_parallel; i++) {
		cout << mp_parallel_shape[i] << ", ";
	}
	cout << ")\n";
	cout << "dim_data: " << m_dim_data << endl;
	cout << "data_shape: (";
	for (int i = 0; i < m_dim_data; i++) {
		cout << mp_data_shape[i] << ", ";
	}
	cout << ")\n";
	cout << "size: " << get_size() << endl;
}

TDD::TDD(weightednode w_node, int dim_parallel, int dim_data, int* p_index_order,
	int64_t* p_parallel_shape, int64_t* p_data_shape) {
	m_wnode = w_node;
	m_dim_parallel = dim_parallel;
	m_dim_data = dim_data;
	mp_index_order = p_index_order;
	mp_parallel_shape = p_parallel_shape;
	mp_data_shape = p_data_shape;
}


TDD::TDD(const TDD& other) {
	m_wnode = other.m_wnode;
	m_dim_parallel = other.m_dim_parallel;
	m_dim_data = other.m_dim_data;
	mp_index_order = array_clone(other.mp_index_order, m_dim_data);
	mp_parallel_shape = array_clone(other.mp_parallel_shape, m_dim_parallel + m_dim_data + 1);
	mp_data_shape = mp_parallel_shape + m_dim_parallel;
}

TDD::~TDD() {
	//mp_data_shape should not be freed because it points into mp_parallel_shape.
#ifdef DECONSTRUCTOR_DEBUG
	if (mp_index_order == nullptr || mp_parallel_shape == nullptr) {
		std::cout << "TDD repeat deconstruction" << std::endl;
	}
	free(mp_index_order);
	mp_index_order = nullptr;
	free(mp_parallel_shape);
	mp_parallel_shape = nullptr;
#elif
	free(mp_index_order);
	free(mp_parallel_shape);
#endif
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

TDD& TDD::operator=(const TDD& other) {
	if (mp_data_shape != nullptr) {
		free(mp_data_shape);
	}
	if (mp_parallel_shape != nullptr) {
		free(mp_parallel_shape);
	}
	if (mp_index_order != nullptr) {
		free(mp_index_order);
	}
	m_wnode = other.m_wnode;
	m_dim_data = other.m_dim_data;
	m_dim_parallel = other.m_dim_parallel;
	mp_data_shape = array_clone(other.mp_data_shape, m_dim_data);
	mp_index_order = array_clone(other.mp_index_order, m_dim_data);
	mp_parallel_shape = array_clone(other.mp_parallel_shape, m_dim_parallel);
	return *this;
}

TDD TDD::clone() const {
	TDD* p_res = new TDD(*this);
	return *p_res;
}

TDD TDD::as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const int* p_index_order) {
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

	//note the ownership transfer here
	TDD* p_res = new TDD(w_node, dim_parallel, dim_data, p_index_order_pd, p_parallel_shape, p_data_shape);
	return *p_res;
}

TDD TDD::as_tensor(const TDD& other) {
	return other.clone();
}

TDD TDD::direct_product(const TDD& a, const TDD& b, bool parallel_tensor) {
	if (parallel_tensor) {
		throw -100;
	}
	else {
		// check the equality of parallel shapes
		//...
	}
	auto w_node = wnode::direct_product(a.m_wnode, a.m_dim_data, b.m_wnode, parallel_tensor);
	//It can be different for parallel situations
	auto dim_parallel = a.m_dim_parallel;
	auto dim_data = a.m_dim_data + b.m_dim_data;
	auto temp_index_order = (int*)malloc(sizeof(int) * (a.m_dim_data + b.m_dim_data));
	for (int i = 0; i < a.m_dim_data; i++) {
		temp_index_order[i] = a.mp_index_order[i];
	}
	for (int i = 0; i < b.m_dim_data; i++) {
		temp_index_order[i + a.m_dim_data] = b.mp_index_order[i] + a.m_dim_data;
	}
	auto p_parallel_shape = (int64_t*)malloc(sizeof(int64_t) * (dim_parallel + dim_data + 1));
	p_parallel_shape[dim_parallel + dim_data] = 2;
	for (int i = 0; i < dim_parallel; i++) {
		p_parallel_shape[i] = a.mp_parallel_shape[i];
	}
	for (int i = 0; i < a.m_dim_data; i++) {
		p_parallel_shape[i+dim_parallel] = a.mp_data_shape[i];
	}	
	for (int i = 0; i < b.m_dim_data; i++) {
		p_parallel_shape[i + dim_parallel+a.m_dim_data] = b.mp_data_shape[i];
	}
	// +1 to put the extra dimension at the end.
	auto p_data_shape = p_parallel_shape + dim_parallel;
	TDD* p_res = new TDD(w_node, dim_parallel, dim_data, temp_index_order, p_parallel_shape, p_data_shape);
	return *p_res;
}

TDD TDD::sum(const TDD& a, const TDD& b) {
	// check whether they are in the same shape
	//...
	auto res_wnode = wnode::sum(a.m_wnode, b.m_wnode);
	TDD&& p_res = a.clone();
	p_res.m_wnode = res_wnode;
	return p_res;
}


CUDAcpl::Tensor TDD::CUDAcpl() const {
	int64_t* p_inner_shape = (int64_t*)malloc(sizeof(int64_t) * (m_dim_data + 1));
	get_inner_data_shape(p_inner_shape);
	CUDAcpl::Tensor res = wnode::to_CUDAcpl(m_wnode, m_dim_data, p_inner_shape);
	free(p_inner_shape);
	return res;
}
