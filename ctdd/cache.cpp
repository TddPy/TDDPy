#include "cache.hpp"

using namespace std;
using namespace weight;
using namespace cache;


unique_table_key<wcomplex>::unique_table_key(int _order, const node::succ_ls<wcomplex>& successors) {
	order = _order;
	code1 = vector<int>(successors.size());
	code2 = vector<int>(successors.size());
	nodes = vector<const node::Node<wcomplex>*>(successors.size());
	for (int i = 0; i < successors.size(); i++) {
		code1[i] = get_int_key(successors[i].weight.real());
		code2[i] = get_int_key(successors[i].weight.imag());
		nodes[i] = successors[i].node;
	}
}



unique_table_key<CUDAcpl::Tensor>::unique_table_key(int _order, const node::succ_ls<CUDAcpl::Tensor>& successors) {
	throw 123456;
}


sum_key<wcomplex>::sum_key(int id_a, const wcomplex& weight_a, int id_b, const wcomplex& weight_b)
{
	if (id_a < id_b) {
		id_1 = id_a;
		id_2 = id_b;
		nweight1_code1 = std::vector<int>({ weight::get_int_key(weight_a.real()) });
		nweight1_code2 = std::vector<int>({ weight::get_int_key(weight_a.imag()) });
		nweight2_code1 = std::vector<int>({ weight::get_int_key(weight_b.real()) });
		nweight2_code2 = std::vector<int>({ weight::get_int_key(weight_b.imag()) });
	}
	else {
		id_1 = id_b;
		id_2 = id_a;
		nweight1_code1 = std::vector<int>({ weight::get_int_key(weight_b.real()) });
		nweight1_code2 = std::vector<int>({ weight::get_int_key(weight_b.imag()) });
		nweight2_code1 = std::vector<int>({ weight::get_int_key(weight_a.real()) });
		nweight2_code2 = std::vector<int>({ weight::get_int_key(weight_a.imag()) });
	}
}




