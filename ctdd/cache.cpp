#include "cache.h"

using namespace weights;
using namespace cache;

unique_table_key<wcomplex>::unique_table_key(node_int _order, node_int _range, const wcomplex* _p_weights, const node::Node<wcomplex>** _p_nodes) {
	order = _order;
	range = _range;
	p_weights_real = (int*)malloc(sizeof(int) * range);
	p_weights_imag = (int*)malloc(sizeof(int) * range);
	for (int i = 0; i < range; i++) {
		p_weights_real[i] = weights::get_int_key(_p_weights[i].real());
		p_weights_imag[i] = weights::get_int_key(_p_weights[i].imag());
	}
	p_nodes = _p_nodes;
}

unique_table_key<CUDAcpl::Tensor>::unique_table_key(node_int _order, node_int _range,
	const CUDAcpl::Tensor* _p_weights, const node::Node<CUDAcpl::Tensor>** _p_nodes) {
	order = _order;
	range = _range;
	auto dim = _p_weights[0].dim();
	auto len = _p_weights[0].numel();
	//p_weights_real is used to store the data, with the first element being the length
	p_weights_real = (int*)malloc(sizeof(int) * (range * len + 1));
	p_weights_real[0] = len;
	//p_weights_imag is used to store the shape, with the first element being the dim
	p_weights_imag = (int*)malloc(sizeof(int) * (dim + 1));
	p_weights_imag[0] = dim;
	//store the data
	for (int i = 0; i < range; i++) {
		auto flattened = _p_weights[i].view({ len });
		for (int j = 0; j < len; j++) {
			p_weights_real[i * len + j + 1] = weights::get_int_key(flattened[i][j].item().toDouble());
		}
	}
	//store the parallel shape
	for (int i = 0; i < dim; i++) {
		p_weights_imag[i + 1] = _p_weights[0].size(i);
	}
	p_nodes = _p_nodes;
}


unique_table_key<wcomplex>::unique_table_key(const unique_table_key<wcomplex>& other) {
	order = other.order;
	range = other.range;
	p_weights_real = array_clone(other.p_weights_real, range);
	p_weights_imag = array_clone(other.p_weights_imag, range);
	p_nodes = other.p_nodes;
}

unique_table_key<CUDAcpl::Tensor>::unique_table_key(const unique_table_key<CUDAcpl::Tensor>& other) {
	order = other.order;
	range = other.range;
	p_weights_real = array_clone(other.p_weights_real, other.p_weights_real[0] + 1);
	p_weights_imag = array_clone(other.p_weights_imag, other.p_weights_imag[0] + 1);
	p_nodes = other.p_nodes;
}


unique_table_key<wcomplex>& unique_table_key<wcomplex>::operator =(const unique_table_key<wcomplex>& other) {
	if (range != other.range) {
		range = other.range;
		free(p_weights_real);
		free(p_weights_imag);
		p_weights_real = (int*)malloc(sizeof(int) * range);
		p_weights_imag = (int*)malloc(sizeof(int) * range);
	}
	order = other.order;
	p_nodes = other.p_nodes;
	for (int i = 0; i < range; i++) {
		p_weights_real[i] = other.p_weights_real[i];
		p_weights_imag[i] = other.p_weights_imag[i];
	}
	return *this;
}

unique_table_key<CUDAcpl::Tensor>& unique_table_key<CUDAcpl::Tensor>::operator =(const unique_table_key<CUDAcpl::Tensor>& other) {
	free(p_weights_real);
	free(p_weights_imag);
	range = other.range;
	order = other.order;
	p_nodes = other.p_nodes;
	p_weights_real = array_clone(other.p_weights_real, other.p_weights_real[0] + 1);
	p_weights_imag = array_clone(other.p_weights_imag, other.p_weights_imag[0] + 1);
	return *this;
}


bool cache::operator==(const unique_table_key<wcomplex>& a, const unique_table_key<wcomplex>& b) {
	if (a.order == b.order &&
		a.range == b.range) {
		for (int i = 0; i < a.range; i++) {
			if (a.p_weights_real[i] == b.p_weights_real[i] &&
				a.p_weights_imag[i] == b.p_weights_imag[i] &&
				a.p_nodes[i] == b.p_nodes[i]) {
				continue;
			}
			else {
				return false;
			}
		}
		return true;
	}
	else {
		return false;
	}
}


bool cache::operator==(const unique_table_key<CUDAcpl::Tensor>& a, const unique_table_key<CUDAcpl::Tensor>& b) {
	if (a.order != b.order ||
		a.range != b.range) {
		return false;
	}
	//check whether they are in the same shape
	if (a.p_weights_imag[0] != b.p_weights_imag[0]) {
		return false;
	}
	for (int i = 0; i < a.p_weights_imag[0]; i++) {
		if (a.p_weights_imag[i + 1] != b.p_weights_imag[i + 1]) {
			return false;
		}
	}
	// check the tensor data
	for (int i = 0; i < a.p_weights_real[0]; i++) {
		if (a.p_weights_real[i + 1] != b.p_weights_real[i + 1]) {
			return false;
		}
	}
	//check the successors
	for (int i = 0; i < a.range; i++) {
		if (a.p_nodes[i] != b.p_nodes[i]) {
			return false;
		}
	}
	return true;
}


std::size_t cache::hash_value(const unique_table_key<wcomplex>& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.order);
	for (int i = 0; i < key_struct.range; i++) {
		boost::hash_combine(seed, key_struct.p_weights_real[i]);
		boost::hash_combine(seed, key_struct.p_weights_imag[i]);
	}
	for (int i = 0; i < key_struct.range; i++) {
		boost::hash_combine(seed, key_struct.p_nodes[i]);
	}
	return seed;
}

std::size_t cache::hash_value(const unique_table_key<CUDAcpl::Tensor>& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.order);
	// combine the data
	for (int i = 0; i < key_struct.p_weights_real[0]; i++) {
		boost::hash_combine(seed, key_struct.p_weights_real[i + 1]);
	}
	// combine the shape
	for (int i = 0; i < key_struct.p_weights_imag[0]; i++) {
		boost::hash_combine(seed, key_struct.p_weights_imag[i + 1]);
	}
	// combine the successors
	for (int i = 0; i < key_struct.range; i++) {
		boost::hash_combine(seed, key_struct.p_nodes[i]);
	}
	return seed;
}


cache::sum_key<wcomplex>::sum_key(int id_a, wcomplex weight_a, int id_b, wcomplex weight_b) {
	if (id_a < id_b) {
		id_1 = id_a;
		id_2 = id_b;
		nweight1_real = weights::get_int_key(weight_a.real());
		nweight1_imag = weights::get_int_key(weight_a.imag());
		nweight2_real = weights::get_int_key(weight_b.real());
		nweight2_imag = weights::get_int_key(weight_b.imag());
	}
	else {
		id_1 = id_b;
		id_2 = id_a;
		nweight1_real = weights::get_int_key(weight_b.real());
		nweight1_imag = weights::get_int_key(weight_b.imag());
		nweight2_real = weights::get_int_key(weight_a.real());
		nweight2_imag = weights::get_int_key(weight_a.imag());
	}
}


bool cache::operator == (const sum_key<wcomplex>& a, const sum_key<wcomplex>& b) {
	return a.id_1 == b.id_1 && a.id_2 == b.id_2 &&
		a.nweight1_real == b.nweight1_real &&
		a.nweight1_imag == b.nweight1_imag &&
		a.nweight2_real == b.nweight2_real &&
		a.nweight2_imag == b.nweight2_imag;
}

std::size_t cache::hash_value(const sum_key<wcomplex>& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.id_1);
	boost::hash_combine(seed, key_struct.nweight1_real);
	boost::hash_combine(seed, key_struct.nweight1_imag);
	boost::hash_combine(seed, key_struct.id_2);
	boost::hash_combine(seed, key_struct.nweight2_real);
	boost::hash_combine(seed, key_struct.nweight2_imag);
	return seed;
}


cache::duplicate_table<wcomplex> cache::Global_Cache<wcomplex>::duplicate_cache = cache::duplicate_table<wcomplex>();
cache::duplicate_table<wcomplex> cache::Global_Cache<wcomplex>::shift_cache = cache::duplicate_table<wcomplex>();
cache::CUDAcpl_table<wcomplex> cache::Global_Cache<wcomplex>::CUDAcpl_cache = cache::CUDAcpl_table<wcomplex>();
cache::sum_table<wcomplex> cache::Global_Cache<wcomplex>::sum_cache = cache::sum_table<wcomplex>();
cache::cont_table<wcomplex> cache::Global_Cache<wcomplex>::cont_cache = cache::cont_table<wcomplex>();

cache::duplicate_table<CUDAcpl::Tensor> cache::Global_Cache<CUDAcpl::Tensor>::duplicate_cache = cache::duplicate_table<CUDAcpl::Tensor>();
cache::duplicate_table<CUDAcpl::Tensor> cache::Global_Cache<CUDAcpl::Tensor>::shift_cache = cache::duplicate_table<CUDAcpl::Tensor>();
cache::CUDAcpl_table<CUDAcpl::Tensor> cache::Global_Cache<CUDAcpl::Tensor>::CUDAcpl_cache = cache::CUDAcpl_table<CUDAcpl::Tensor>();
cache::sum_table<CUDAcpl::Tensor> cache::Global_Cache<CUDAcpl::Tensor>::sum_cache = cache::sum_table<CUDAcpl::Tensor>();
cache::cont_table<CUDAcpl::Tensor> cache::Global_Cache<CUDAcpl::Tensor>::cont_cache = cache::cont_table<CUDAcpl::Tensor>();
