#include "weighted_node.h"
#include "weights.h"

using namespace std;
using namespace CUDAcpl;
using namespace node;
using namespace wnode;


dict::sum_key::sum_key(int id_a, wcomplex weight_a, int id_b, wcomplex weight_b) {
	if (id_a < id_b) {
		id_1 = id_a;
		id_2 = id_b;
		nweight1_real = Node::get_int_key(weight_a.real());
		nweight1_imag = Node::get_int_key(weight_a.imag());
		nweight2_real = Node::get_int_key(weight_b.real());
		nweight2_imag = Node::get_int_key(weight_b.imag());
	}
	else {
		id_1 = id_b;
		id_2 = id_a;
		nweight1_real = Node::get_int_key(weight_b.real());
		nweight1_imag = Node::get_int_key(weight_b.imag());
		nweight2_real = Node::get_int_key(weight_a.real());
		nweight2_imag = Node::get_int_key(weight_a.imag());
	}
}


bool dict::operator == (const sum_key& a, const sum_key& b) {
	return a.id_1 == b.id_1 && a.id_2 == b.id_2 &&
		a.nweight1_real == b.nweight1_real &&
		a.nweight1_imag == b.nweight1_imag &&
		a.nweight2_real == b.nweight2_real &&
		a.nweight2_imag == b.nweight2_imag;
}

std::size_t dict::hash_value(const sum_key& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.id_1);
	boost::hash_combine(seed, key_struct.nweight1_real);
	boost::hash_combine(seed, key_struct.nweight1_imag);
	boost::hash_combine(seed, key_struct.id_2);
	boost::hash_combine(seed, key_struct.nweight2_real);
	boost::hash_combine(seed, key_struct.nweight2_imag);
	return seed;
}




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
	const Node** p_new_successors = (const Node**)malloc(sizeof(const Node*) * the_successors.size());
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
	const Node** new_successors = (const Node**)malloc(sizeof(const Node*) * range);
	for (int i = 0; i < range; i++) {
		new_weights[i] = weights[i] / weig_max;
		new_successors[i] = successors[i];
	}
	const Node* new_node = Node::get_unique_node(w_node.p_node->get_order(), range, new_weights, new_successors);
	weightednode res = weightednode{ weig_max * w_node.weight, new_node };
	return res;
}

CUDAcpl::Tensor wnode::to_CUDAcpl_iterate(weightednode w_node, int dim_data, int64_t* p_data_shape, dict::CUDAcpl_table& tensor_cache) {
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
			auto key = Node::get_id_all(succ);
			auto p_res = tensor_cache.find(key);
			if (p_res != tensor_cache.end()) {
				uniform_tensor = p_res->second;
			}
			else {
				auto next_wnode = weightednode{ wcomplex(1.,0.), succ};
				uniform_tensor = to_CUDAcpl_iterate(next_wnode, dim_data, p_data_shape, tensor_cache);
				//add into the dictionary
				tensor_cache.insert(std::make_pair(key, uniform_tensor));
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


weightednode wnode::direct_product(weightednode a, int a_depth, weightednode b, bool parallel_tensor) {
	wcomplex weight;
	if (parallel_tensor) {
		//not implemented yet
		throw - 1;
	}
	else {
		weight = a.weight * b.weight;
	}
	auto p_res_node = Node::append(a.p_node, a_depth, b.p_node, parallel_tensor);
	weightednode res = weightednode{ weight, p_res_node };
	return res;
}

sum_nweights wnode::sum_weights_normalize(wcomplex weight1, wcomplex weight2) {
	sum_nweights res = sum_nweights{ wcomplex(0.,0.), wcomplex(0.,0.), wcomplex(1.,0.) };
	//chose the larger norm
	wcomplex renorm_coef = (norm(weight1) > norm(weight2)) ? weight1 : weight2;
	if (norm(renorm_coef) > Node::EPS()) {
		res.nweight1 = weight1 / renorm_coef;
		res.nweight2 = weight2 / renorm_coef;
		res.renorm_coef = renorm_coef;
	}
	return res;
}


weightednode wnode::sum_iterate(weightednode w_node1, weightednode w_node2, wcomplex renorm_coef, dict::sum_table sum_cache) {
	if (w_node1.p_node == TERMINAL_NODE && w_node2.p_node == TERMINAL_NODE) {
		weightednode res = weightednode{ (w_node1.weight + w_node2.weight) * renorm_coef, TERMINAL_NODE };
		return res;
	}
	
	// produce the unique key and look up in the dictionary
	auto key = dict::sum_key(Node::get_id_all(w_node1.p_node), w_node1.weight,
		Node::get_id_all(w_node2.p_node), w_node2.weight);

	auto p_find_res = sum_cache.find(key);
	if (p_find_res != sum_cache.end()) {
		weightednode res = p_find_res->second;
		res.weight = res.weight * renorm_coef;
		return res;
	}
	else {
		node_int range = 0;
		if (w_node1.p_node != TERMINAL_NODE) {
			range = w_node1.p_node->get_range();
		}
		else {
			range = w_node2.p_node->get_range();
		}
		wcomplex* p_weights = (wcomplex*)malloc(sizeof(wcomplex) * range);
		const Node** p_nodes = (const Node**)malloc(sizeof(const Node*) * range);

		// A and B are used to refer to the nodes of smaller and larger orders.
		weightednode A, B;

		if (w_node1.p_node != TERMINAL_NODE && w_node2.p_node != TERMINAL_NODE &&
			w_node1.p_node->get_order() == w_node2.p_node->get_order()) {
			auto successors1 = w_node1.p_node->get_successors();
			auto successors2 = w_node2.p_node->get_successors();
			for (int i = 0; i < range; i++) {
				auto next_weight1 = w_node1.weight * w_node1.p_node->get_weights()[i];
				auto next_weight2 = w_node2.weight * w_node2.p_node->get_weights()[i];
				//normalize as a whole
				auto renorm_res = sum_weights_normalize(next_weight1, next_weight2);
				weightednode next_wnode1 = weightednode{ renorm_res.nweight1, successors1[i] };
				weightednode next_wnode2 = weightednode{ renorm_res.nweight2, successors2[i] };
				weightednode res = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef, sum_cache);
				p_nodes[i] = res.p_node;
				p_weights[i] = res.weight;
			}
			A = w_node1;
		}
		else {
			/*  
				There are three cases following, corresponding to the same procedure:
                1. node1 == None, node2 != None
                2. node2 == None, node1 != None
                3. node1 != None, node2 != None, but node1.order != noder2.order
                We first analysis the situation to reuse the codes.
                A will be the lower ordered weighted node.
			*/
			if (w_node1.p_node == TERMINAL_NODE) {
				A = w_node2;
				B = w_node1;
			}
			else if (w_node2.p_node == TERMINAL_NODE) {
				A = w_node1;
				B = w_node2;
			}
			else if (w_node1.p_node->get_order() < w_node2.p_node->get_order()) {
				A = w_node1;
				B = w_node2;
			}
			else {
				A = w_node2;
				B = w_node1;
			}
			auto weightsA = A.p_node->get_weights();
			auto successorsA = A.p_node->get_successors();
			for (int i = 0; i < range; i++) {
				auto next_weight_A = A.weight * weightsA[i];
				//normalize as a whole
				auto renorm_res = sum_weights_normalize(next_weight_A, B.weight);
				weightednode next_wnodeA = weightednode{ renorm_res.nweight1, successorsA[i] };
				weightednode next_wnodeB = weightednode{ renorm_res.nweight2, B.p_node };
				weightednode res = sum_iterate(next_wnodeA, next_wnodeB, renorm_res.renorm_coef, sum_cache);
				p_weights[i] = res.weight;
				p_nodes[i] = res.p_node;
			}
		}
		auto temp_new_node = Node(0, A.p_node->get_order(), range, p_weights, p_nodes);
		auto res = normalize(weightednode{ wcomplex(1.,0.), &temp_new_node });

		//cache the result
		sum_cache.insert(std::make_pair(key, res));
		
		//finally multiply the renorm_coef
		res.weight = res.weight * renorm_coef;
		return res;
	}

}

weightednode wnode::sum(weightednode w_node1, weightednode w_node2, dict::sum_table* p_sum_cache) {
	dict::sum_table* p_cache;
	if (p_sum_cache == nullptr) {
		p_cache = new dict::sum_table();
	}
	else {
		p_cache = p_sum_cache;
	}
	// normalize as a whole
	auto renorm_res = sum_weights_normalize(w_node1.weight, w_node2.weight);
	weightednode next_wnode1 = weightednode{ renorm_res.nweight1, w_node1.p_node };
	weightednode next_wnode2 = weightednode{ renorm_res.nweight2, w_node2.p_node };
	auto res = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef, *p_cache);
	if (p_sum_cache == nullptr) {
		delete p_cache;
	}
	return res;
}

