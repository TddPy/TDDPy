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

std::pair<int64_t*, int*> TDD::index_reduced_info(int length, int* p_inner_i) const {
	int* p_order = (int*)malloc(sizeof(int) * (m_dim_data - length));
	int64_t* p_shape = (int64_t*)malloc(sizeof(int64_t) * (m_dim_data + m_dim_parallel + 1 - length));
	p_shape[m_dim_data + m_dim_parallel - length] = 2;
	for (int i = 0; i < m_dim_parallel; i++) {
		p_shape[i] = mp_parallel_shape[i];
	}

	auto p_inner_shape = inner_data_shape();

	int new_count = 0;
	for (int i = 0; i < m_dim_data; i++) {
		// check whether this inner index is reduced
		bool reduced = false;
		for (int j = 0; j < length; j++) {
			if (i == p_inner_i[j]) {
				reduced = true;
				break;
			}
		}
		if (!reduced) {
			p_shape[new_count + m_dim_parallel] = p_inner_shape[i];
			p_order[new_count] = mp_index_order[i];
			new_count++;
		}
	}

	free(p_inner_shape);

	int* p_order_res = (int*)malloc(sizeof(int) * (m_dim_data - length));
	for (int i = 0; i < m_dim_data - length; i++) {
		p_order_res[i] = i;
	}
	sort(p_order_res, p_order_res + m_dim_data - length,
		[p_order](int a, int b) {
			return (p_order[a] < p_order[b]);
		}
	);
	int64_t* p_shape_res = array_clone(p_shape, m_dim_data + m_dim_parallel + 1 - length);
	//sort the data_shape
	for (int i = 0; i < m_dim_data - length; i++) {
		p_shape_res[m_dim_parallel + i] = p_shape[p_order_res[i]];
	}
	free(p_order);
	free(p_shape);
	return std::make_pair(p_shape_res, p_order_res);
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
	int64_t* p_parallel_shape) {
	m_wnode = w_node;
	m_dim_parallel = dim_parallel;
	m_dim_data = dim_data;
	mp_index_order = p_index_order;
	mp_parallel_shape = p_parallel_shape;
	mp_data_shape = p_parallel_shape + m_dim_parallel;
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

int* TDD::inversed_order() const {
	int* p_res = (int*)malloc(sizeof(int) * m_dim_data);
	for (int i = 0; i < m_dim_data; i++) {
		p_res[mp_index_order[i]] = i;
	}
	return p_res;
}

int64_t* TDD::inner_data_shape() const {
	int64_t* p_res = (int64_t*)malloc(sizeof(int64_t) * m_dim_data);
	for (int i = 0; i < m_dim_data; i++) {
		p_res[i] = mp_data_shape[mp_index_order[i]];
	}
	return p_res;
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
	TDD* p_res = new TDD(w_node, dim_parallel, dim_data, p_index_order_pd, p_parallel_shape);
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
	TDD* p_res = new TDD(w_node, dim_parallel, dim_data, temp_index_order, p_parallel_shape);
	return *p_res;
}

TDD TDD::sum(const TDD& a, const TDD& b) {
	// check whether they are in the same shape
	//...
	dict::sum_table* p_cache = new dict::sum_table();
	auto res_wnode = wnode::sum(a.m_wnode, b.m_wnode, p_cache);
	TDD&& res = a.clone();
	res.m_wnode = res_wnode;
	delete p_cache;
	return res;
}

TDD TDD::contract(int num_pair, const int* p_i1, const int* p_i2, dict::sum_table* p_sum_cache) {
	if (num_pair == 0) {
		return clone();
	}
	else {
		//transform to inner indices
		auto p_inversed_order = inversed_order();
		auto p_inner_data_shape = inner_data_shape();
		int* p_inner_i1 = (int*)malloc(sizeof(int) * num_pair);
		int* p_inner_i2 = (int*)malloc(sizeof(int) * num_pair);
		for (int i = 0; i < num_pair; i++) {
			p_inner_i1[i] = p_inversed_order[p_i1[i]];
			p_inner_i2[i] = p_inversed_order[p_i2[i]];
		}

		//note that inner_ls1[i] < inner_ls2[i] should hold for every i.
		auto&& res_wnode = wnode::contract(m_wnode, m_dim_data, p_inner_data_shape, num_pair, p_inner_i1, p_inner_i2, p_sum_cache);
		
		auto p_inner_i_reduced = array_concat(p_inner_i1, num_pair, p_inner_i2, num_pair);
		auto reduced_info = index_reduced_info(2 * num_pair, p_inner_i_reduced);
		free(p_inner_i_reduced);

		auto&& res_tdd = TDD(res_wnode, m_dim_parallel, m_dim_data - 2 * num_pair, reduced_info.second, reduced_info.first);

		free(p_inversed_order);
		free(p_inner_data_shape);
		free(p_inner_i1);
		free(p_inner_i2);
		return res_tdd;
	}
}


TDD TDD::tensordot(const TDD& a, const TDD& b, int num_pair, const int* p_ia, const int* p_ib, bool parallel_tensor) {
	auto&& temp_tdd = direct_product(a, b, parallel_tensor);
	int* ptemp_ib = (int*)malloc(sizeof(int) * num_pair);
	for (int i = 0; i < num_pair; i++) {
		ptemp_ib[i] = p_ib[i] + a.m_dim_data;
	}
	auto&& res = temp_tdd.contract(num_pair, p_ia, ptemp_ib, nullptr);
	free(ptemp_ib);
	return res;
}

TDD TDD::tensordot(const TDD& a, const TDD& b, int num_indices, bool parallel_tensor) {
	int* p_ia = (int*)malloc(sizeof(int) * num_indices);
	int* p_ib = (int*)malloc(sizeof(int) * num_indices);
	for (int i = 0; i < num_indices; i++) {
		p_ia[i] = a.m_dim_data - num_indices + i;
		p_ib[i] = i;
	}
	TDD&& res = tensordot(a, b, num_indices, p_ia, p_ib, parallel_tensor);
	free(p_ia);
	free(p_ib);
	return res;
}



CUDAcpl::Tensor TDD::CUDAcpl() const {
	int64_t* p_inner_shape = (int64_t*)malloc(sizeof(int64_t) * (m_dim_data + 1));
	get_inner_data_shape(p_inner_shape);
	CUDAcpl::Tensor res = wnode::to_CUDAcpl(m_wnode, m_dim_data, p_inner_shape);
	free(p_inner_shape);
	return res;
}
