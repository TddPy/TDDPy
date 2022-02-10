#include "weighted_node.h"
#include "weights.h"

using namespace std;
using namespace CUDAcpl;
using namespace node;
using namespace wnode;

inline bool wnode::is_equal(weightednode a, weightednode b) {
	return a.p_node == b.p_node && weights::is_equal(a.weight, b.weight, Node::EPS());
}

weightednode wnode::as_tensor_iterate(const CUDAcpl::Tensor& t,
	int dim_parallel, const int64_t* p_parallel_shape,
	int dim_data, const int64_t* p_data_shape, const int* p_index_order, int depth) {
	//checks whether the tensor is reduced to the [[...[val]...]] form
	weightednode res;
	if (depth == dim_data) {
		if (dim_data == 0) {
			res.weight = item(t);
		}
		else {
			res.weight = item(t);
		}
		res.p_node = node::TERMINAL_NODE;
		return res;
	}

	int split_pos = p_index_order[depth];
	auto split_tensor = t.split(1, t.dim() - dim_data + split_pos - 1);
	//-1 is because the extra inner dim for real and imag
	
	vector<weightednode> the_successors = vector<weightednode>();

	for (auto tensor = split_tensor.begin(); tensor != split_tensor.end(); tensor++) {
		the_successors.push_back(as_tensor_iterate(*tensor, dim_parallel, 
			p_parallel_shape, dim_data, p_data_shape, p_index_order, depth + 1));
	}

	// stack the weighted subnodes
	wcomplex* p_new_weights = (wcomplex*)malloc(sizeof(wcomplex) * the_successors.size());
	const Node** p_new_successors = (const Node**)malloc(sizeof(Node*) * the_successors.size());
	for (int i = 0; i < the_successors.size(); i++) {
		p_new_weights[i] = the_successors[i].weight;
		p_new_successors[i] = the_successors[i].p_node;
	}
	Node temp_node = Node(0, depth, the_successors.size(), p_new_weights, p_new_successors);
	//normalize this depth
	weightednode new_wnode = weightednode{ wcomplex(1.,0.), &temp_node };
	return normalize(new_wnode);
}


weightednode wnode::normalize(weightednode w_node) {
	if (w_node.p_node == TERMINAL_NODE) {
		return w_node;
	}

	// redirect zero weighted nodes to the terminal node
	if (abs(w_node.weight.real()) < Node::EPS() &&
		abs(w_node.weight.imag()) < Node::EPS()) {
		weightednode res = { wcomplex(0.,0.), TERMINAL_NODE };
		return res;
	}

	node_int range = w_node.p_node->get_range();
	auto successors = w_node.p_node->get_successors();
	auto weights = w_node.p_node->get_weights();
	// node reduction check (reduce when all equal)
	bool all_equal = true;
	auto wnode_0 = weightednode{ weights[0], successors[0] };
	for (int i = 1; i < range; i++) {
		auto wnode_i = weightednode{ weights[i], successors[i] };
		if (!is_equal(wnode_0, wnode_i)) {
			all_equal = false;
			break;
		}
	}
	if (all_equal) {
		wcomplex new_weight = w_node.weight * weights[0];
		weightednode res = weightednode{ new_weight, successors[0] };
		return res;
	}

	// check whether all successor weights are zero, and redirect to terminal node if so
	bool all_zero = true;
	for (int i = 0; i < range; i++) {
		if (abs(weights[i].real()) > Node::EPS() ||
			abs(weights[i].imag()) > Node::EPS()) {
			all_zero = false;
			break;
		}
	}
	if (all_zero) {
		weightednode res = weightednode{ wcomplex(0.,0.), TERMINAL_NODE };
		return res;
	}

	// start to normalize the weights
	int i_max = 0;
	double norm_max = norm(weights[0]);
	for (int i = 1; i < range; i++) {
		double temp_norm = norm(weights[i]);
		if (temp_norm > norm_max) {
			i_max = i;
			norm_max = temp_norm;
		}
	}
	wcomplex weig_max = weights[i_max];
	wcomplex* new_weights = (wcomplex*)malloc(sizeof(wcomplex) * range);
	const Node** new_successors = (const Node**)malloc(sizeof(Node*) * range);
	for (int i = 0; i < range; i++) {
		new_weights[i] = weights[i] / weig_max;
		new_successors[i] = successors[i];
	}
	Node* new_node = Node::get_unique_node(w_node.p_node->get_order(), range, new_weights, new_successors);
	weightednode res = weightednode{ weig_max * w_node.weight, new_node };
	return res;
}

CUDAcpl::Tensor wnode::to_CUDAcpl_iterate(weightednode w_node, int dim_data, int64_t* p_data_shape, dict::CUDAcpl_table& tensor_dict) {
	// w_node.p_node is guaranteed not to be the TERMINAL_NODE

	node_int current_order = w_node.p_node->get_order();

	auto par_tensor = vector<CUDAcpl::Tensor>();
	auto p_weights = w_node.p_node->get_weights();
	auto p_successors = w_node.p_node->get_successors();

	// The temp shape for adjustment.
	auto new_len = dim_data - current_order;
	int64_t* p_temp_shape = (int64_t*)malloc(sizeof(int64_t) * new_len);
	p_temp_shape[new_len-1] = 2;

	for (int k = 0; k < w_node.p_node->get_range(); k++) {
		// detect terminal nodes, or iterate on the next node
		auto succ = p_successors[k];
		auto weight = p_weights[k];
		CUDAcpl::Tensor temp_tensor;
		CUDAcpl::Tensor uniform_tensor;
		int next_order;
		if (succ == node::TERMINAL_NODE) {
			temp_tensor = CUDAcpl::from_complex(weight);
			next_order = dim_data;
		}
		else {
			// first look up in the dictionary
			auto key = succ->get_key_struct();
			auto p_res = tensor_dict.find(key);
			if (p_res != tensor_dict.end()) {
				uniform_tensor = p_res->second;
			}
			else {
				auto next_wnode = weightednode{ wcomplex(1.,0.), succ};
				uniform_tensor = to_CUDAcpl_iterate(next_wnode, dim_data, p_data_shape, tensor_dict);
				//add into the dictionary
				tensor_dict[key] = uniform_tensor;
			}
			next_order = succ->get_order();
			// multiply the dangling weight
			temp_tensor = CUDAcpl::mul_element_wise(uniform_tensor, weight);
		}
		//broadcast according to the index distance
		if (next_order - current_order > 1) {
			//prepare the new data shape
			for (int i = 0; i < next_order - current_order - 1; i++) {
				p_temp_shape[i] = 1;
			}
			for (int i = 0; i < temp_tensor.dim() - 1; i++) {
				p_temp_shape[i + next_order - current_order - 1] = temp_tensor.size(i);
			}
			auto indices = c10::IntArrayRef(p_temp_shape, new_len);
			temp_tensor = temp_tensor.view(indices);
			for (int i = 0; i < next_order - current_order - 1; i++) {
				p_temp_shape[i] = p_data_shape[i + current_order + 1];
			}
			temp_tensor = temp_tensor.expand(c10::IntArrayRef(p_temp_shape, new_len));
		}
		par_tensor.push_back(temp_tensor);
	}

	free(p_temp_shape);

	auto res = torch::stack(par_tensor, 0);
	// multiply the dangling weight at then end
	res = CUDAcpl::mul_element_wise(res, w_node.weight);
	return res;
}

CUDAcpl::Tensor wnode::to_CUDAcpl(weightednode w_node, int dim_data, int64_t* p_inner_data_shape) {
	CUDAcpl::Tensor res;
	int n_extra_one = 0;
	dict::CUDAcpl_table tensor_dict;
	if (w_node.p_node == node::TERMINAL_NODE) {
		res = CUDAcpl::ones({});
	}
	else {
		tensor_dict = dict::CUDAcpl_table();
		res = to_CUDAcpl_iterate(w_node, dim_data, p_inner_data_shape, tensor_dict);
		n_extra_one = w_node.p_node->get_order();
	}

	// this extra layer is for adding the reduced dimensions at the front
	// prepare the real data shape
	int64_t* p_full_data_shape = (int64_t*)malloc(sizeof(int64_t) * (dim_data + 1));
	p_full_data_shape[dim_data] = 2;
	auto cur_data_shape = res.sizes();
	for (int i = 0; i < n_extra_one; i++) {
		p_full_data_shape[i] = 1;
	}
	for (int i = n_extra_one; i < dim_data; i++) {
		p_full_data_shape[i] = cur_data_shape[i];
	}
	auto full_data_shape = c10::IntArrayRef(p_full_data_shape, dim_data + 1);
	free(p_full_data_shape);
	res = res.view(full_data_shape);
	full_data_shape = c10::IntArrayRef(p_inner_data_shape, dim_data + 1);
	res = res.expand(full_data_shape);
	return res;
}
