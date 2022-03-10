#include "cache.hpp"

using namespace std;
using namespace weight;
using namespace cache;


unique_table_key<wcomplex>::unique_table_key(int _order, const node::succ_ls<wcomplex>& successors) {
	order = _order;
	code1 = vector<WCode>(successors.size());
	code2 = vector<WCode>(successors.size());
	nodes = vector<const node::Node<wcomplex>*>(successors.size());
	for (int i = 0; i < successors.size(); i++) {
		get_int_key(code1.data() + i, successors[i].weight.real());
		get_int_key(code2.data() + i, successors[i].weight.imag());
		nodes[i] = successors[i].node;
	}
}



unique_table_key<CUDAcpl::Tensor>::unique_table_key(int _order, const node::succ_ls<CUDAcpl::Tensor>& successors) {
	order = _order;
	auto numel = successors[0].weight.numel();
	code1 = vector<WCode>(numel * successors.size());
	nodes = vector<const node::Node<CUDAcpl::Tensor>*>(successors.size());
	for (int i = 0; i < successors.size(); i++) {
		get_int_key(code1.data() + i * numel, successors[i].weight);
		nodes[i] = successors[i].node;
	}
	code2 = vector<WCode>(successors[0].weight.dim());
	auto&& sizes = successors[0].weight.sizes();
	for (int i = 0; i < successors[0].weight.dim(); i++) {
		code2[i] = sizes[i];
	}
}


sum_key<wcomplex>::sum_key(int id_a, const wcomplex& weight_a, int id_b, const wcomplex& weight_b)
{
	if (id_a < id_b) {
		id_1 = id_a;
		id_2 = id_b;
		nweight1_code1 = std::vector<WCode>(1);
		get_int_key(nweight1_code1.data(), weight_a.real());
		nweight1_code2 = std::vector<WCode>(1);
		get_int_key(nweight1_code2.data(), weight_a.imag());
		nweight2_code1 = std::vector<WCode>(1);
		get_int_key(nweight2_code1.data(), weight_b.real());
		nweight2_code2 = std::vector<WCode>(1);
		get_int_key(nweight2_code2.data(), weight_b.imag());
	}
	else {
		id_1 = id_b;
		id_2 = id_a;
		nweight1_code1 = std::vector<WCode>(1);
		get_int_key(nweight1_code1.data(), weight_b.real());
		nweight1_code2 = std::vector<WCode>(1);
		get_int_key(nweight1_code2.data(), weight_b.imag());
		nweight2_code1 = std::vector<WCode>(1);
		get_int_key(nweight2_code1.data(), weight_a.real());
		nweight2_code2 = std::vector<WCode>(1);
		get_int_key(nweight2_code2.data(), weight_a.imag());
	}
}

sum_key<CUDAcpl::Tensor>::sum_key(int id_a, const CUDAcpl::Tensor& weight_a, int id_b, const CUDAcpl::Tensor& weight_b) {
	if (id_a < id_b) {
		id_1 = id_a;
		id_2 = id_b;
		nweight1_code1 = std::vector<WCode>(weight_a.numel());
		get_int_key(nweight1_code1.data(), weight_a);

		// store the shape
		nweight1_code2 = std::vector<WCode>(weight_a.dim());
		auto&& sizes = weight_a.sizes();
		for (int i = 0; i < weight_a.dim(); i++) {
			nweight1_code2[i] = sizes[i];
		}

		nweight2_code1 = std::vector<WCode>(weight_b.numel());
		get_int_key(nweight2_code1.data(), weight_b);

		// nweight2_code2 is not needed
		nweight2_code2 = std::vector<WCode>();
	}
	else {
		id_1 = id_b;
		id_2 = id_a;
		nweight1_code1 = std::vector<WCode>(weight_b.numel());
		get_int_key(nweight1_code1.data(), weight_b);

		// store the shape
		nweight1_code2 = std::vector<WCode>(weight_b.dim());
		auto&& sizes = weight_b.sizes();
		for (int i = 0; i < weight_b.dim(); i++) {
			nweight1_code2[i] = sizes[i];
		}

		nweight2_code1 = std::vector<WCode>(weight_a.numel());
		get_int_key(nweight2_code1.data(), weight_a);

		nweight2_code2 = std::vector<WCode>();
	}
}




