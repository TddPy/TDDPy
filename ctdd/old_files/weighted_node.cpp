#include "weighted_node.h"
#include "weights.h"

using namespace std;
using namespace CUDAcpl;
using namespace node;
using namespace wnode;


dict::sum_key<wcomplex>::sum_key(int id_a, wcomplex weight_a, int id_b, wcomplex weight_b) {
	if (id_a < id_b) {
		id_1 = id_a;
		id_2 = id_b;
		nweight1_real = node::Node<wcomplex>::get_int_key(weight_a.real());
		nweight1_imag = node::Node<wcomplex>::get_int_key(weight_a.imag());
		nweight2_real = node::Node<wcomplex>::get_int_key(weight_b.real());
		nweight2_imag = node::Node<wcomplex>::get_int_key(weight_b.imag());
	}
	else {
		id_1 = id_b;
		id_2 = id_a;
		nweight1_real = node::Node<wcomplex>::get_int_key(weight_b.real());
		nweight1_imag = node::Node<wcomplex>::get_int_key(weight_b.imag());
		nweight2_real = node::Node<wcomplex>::get_int_key(weight_a.real());
		nweight2_imag = node::Node<wcomplex>::get_int_key(weight_a.imag());
	}
}


bool dict::operator == (const sum_key<wcomplex>& a, const sum_key<wcomplex>& b) {
	return a.id_1 == b.id_1 && a.id_2 == b.id_2 &&
		a.nweight1_real == b.nweight1_real &&
		a.nweight1_imag == b.nweight1_imag &&
		a.nweight2_real == b.nweight2_real &&
		a.nweight2_imag == b.nweight2_imag;
}

std::size_t dict::hash_value(const sum_key<wcomplex>& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.id_1);
	boost::hash_combine(seed, key_struct.nweight1_real);
	boost::hash_combine(seed, key_struct.nweight1_imag);
	boost::hash_combine(seed, key_struct.id_2);
	boost::hash_combine(seed, key_struct.nweight2_real);
	boost::hash_combine(seed, key_struct.nweight2_imag);
	return seed;
}


dict::CUDAcpl_table<wcomplex> dict::global_CUDAcpl_cache_w = dict::CUDAcpl_table<wcomplex>();
dict::sum_table<wcomplex> dict::global_sum_cache_w = dict::sum_table<wcomplex>();
dict::cont_table<wcomplex> dict::global_cont_cache_w = dict::cont_table<wcomplex>();
