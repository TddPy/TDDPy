#include "Node.h"
#include "weights.h"

using namespace dict;
using namespace node;

unique_table_key<wcomplex>::unique_table_key(node_int _order, node_int _range, const wcomplex* _p_weights, const node::Node<wcomplex>** _p_nodes) {
	order = _order;
	range = _range;
	p_weights_real = (int*)malloc(sizeof(int) * range);
	p_weights_imag = (int*)malloc(sizeof(int) * range);
	for (int i = 0; i < range; i++) {
		p_weights_real[i] = node::Node<wcomplex>::get_int_key(_p_weights[i].real());
		p_weights_imag[i] = node::Node<wcomplex>::get_int_key(_p_weights[i].imag());
	}
	p_nodes = _p_nodes;
}


bool dict::operator==(const unique_table_key<wcomplex>& a, const unique_table_key<wcomplex>& b) {
	if (a.order == b.order &&
		a.range == b.range) {
		double eps = weights::EPS;
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


std::size_t dict::hash_value(const unique_table_key<wcomplex>& key_struct) {
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


dict::duplicate_table<wcomplex> dict::global_duplicate_cache_w = dict::duplicate_table<wcomplex>();
dict::duplicate_table<wcomplex> dict::global_shift_cache_w = dict::duplicate_table<wcomplex>();


/*
	Implementations of Node.
*/

int node::Node<wcomplex>::m_global_id = 0;
unique_table<wcomplex> node::Node<wcomplex>::m_unique_table = unique_table<wcomplex>();
