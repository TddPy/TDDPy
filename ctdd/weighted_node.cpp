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
	int dim_parallel, const int* p_parallel_shape,
	int dim_data, const int* p_data_shape, const int* p_index_order, int depth) {
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
