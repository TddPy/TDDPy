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


dict::cont_key::cont_key(int _id, int _num_remained, const int* _p_r_i1, const int* _p_r_i2,
	int _num_waiting, const int* _p_w_i, const int* _p_w_v) {
	id = _id;
	num_remained = _num_remained;
	p_r_i1 = array_clone(_p_r_i1, num_remained);
	p_r_i2 = array_clone(_p_r_i2, num_remained);
	num_waiting = _num_waiting;
	p_w_i = array_clone(_p_w_i, num_waiting);
	p_w_v = array_clone(_p_w_v, num_waiting);
}

dict::cont_key::cont_key(const cont_key& other) {
	id = other.id;
	num_remained = other.num_remained;
	p_r_i1 = array_clone(other.p_r_i1, num_remained);
	p_r_i2 = array_clone(other.p_r_i2, num_remained);
	num_waiting = other.num_waiting;
	p_w_i = array_clone(other.p_w_i, num_waiting);
	p_w_v = array_clone(other.p_w_v, num_waiting);
}

dict::cont_key& dict::cont_key::operator =(const cont_key& other) {
	id = other.id;
	if (num_remained != other.num_remained) {
		num_remained = other.num_remained;
		free(p_r_i1);
		free(p_r_i2);
		p_r_i1 = (int*)malloc(sizeof(int) * num_remained);
		p_r_i2 = (int*)malloc(sizeof(int) * num_remained);
	}
	if (num_waiting != other.num_waiting) {
		num_waiting = other.num_waiting;
		free(p_w_i);
		free(p_w_v);
		p_w_i = (int*)malloc(sizeof(int) * num_waiting);
		p_w_v = (int*)malloc(sizeof(int) * num_waiting);
	}
	for (int i = 0; i < num_remained; i++) {
		p_r_i1[i] = other.p_r_i1[i];
		p_r_i2[i] = other.p_r_i2[i];
	}
	for (int i = 0; i < num_waiting; i++) {
		p_w_i[i] = other.p_w_i[i];
		p_w_v[i] = other.p_w_v[i];
	}
	return *this;
}

dict::cont_key::~cont_key() {
#ifdef DECONSTRUCTOR_DEBUG
	if (p_r_i1 == nullptr || p_r_i2 == nullptr ||
		p_w_i == nullptr || p_w_v == nullptr) {
		std::cout << "cont_key repeat deconstruction" << std::endl;
	}
	free(p_r_i1);
	p_r_i1 = nullptr;
	free(p_r_i2);
	p_r_i2 = nullptr;
	free(p_w_i);
	p_w_i = nullptr;
	free(p_w_v);
	p_w_v = nullptr;
#elif
	free(p_r_i1);
	free(p_r_i2);
	free(p_w_i);
	free(p_w_v);
#endif
}

bool dict::operator == (const cont_key& a, const cont_key& b) {
	if (a.id != b.id || a.num_remained != b.num_remained || a.num_waiting != b.num_waiting) {
		return false;
	}
	for (int i = 0; i < a.num_remained; i++) {
		if (a.p_r_i1[i] != b.p_r_i1[i] || a.p_r_i2[i] != b.p_r_i2[i]) {
			return false;
		}
	}
	for (int i = 0; i < a.num_waiting; i++) {
		if (a.p_w_i[i] != b.p_w_i[i] || a.p_w_v[i] != b.p_w_v[i]) {
			return false;
		}
	}
	return true;
}


std::size_t dict::hash_value(const cont_key& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.id);
	for (int i = 0; i < key_struct.num_remained; i++) {
		boost::hash_combine(seed, key_struct.p_r_i1[i]);
		boost::hash_combine(seed, key_struct.p_r_i2[i]);
	}
	for (int i = 0; i < key_struct.num_waiting; i++) {
		boost::hash_combine(seed, key_struct.p_w_i[i]);
		boost::hash_combine(seed, key_struct.p_w_v[i]);
	}
	return seed;
}


dict::CUDAcpl_table dict::global_CUDAcpl_cache = dict::CUDAcpl_table();
dict::sum_table dict::global_sum_cache = dict::sum_table();
dict::cont_table dict::global_cont_cache = dict::cont_table();

/*
* implementation of wnode
*/


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
	Node&& temp_node = Node(0, depth, the_successors.size(), p_new_weights, p_new_successors);
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
	p_temp_shape[new_len - 1] = 2;

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
				auto next_wnode = weightednode{ wcomplex(1.,0.), succ };
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
		res = CUDAcpl::mul_element_wise(CUDAcpl::ones({}), w_node.weight);
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
		auto&& temp_new_node = Node(0, A.p_node->get_order(), range, p_weights, p_nodes);
		auto res = normalize(weightednode{ wcomplex(1.,0.), &temp_new_node });

		//cache the result
		sum_cache.insert(std::make_pair(key, res));

		//finally multiply the renorm_coef
		res.weight = res.weight * renorm_coef;
		return res;
	}

}

weightednode wnode::sum(weightednode w_node1, weightednode w_node2, dict::sum_table* p_sum_cache) {
	if (p_sum_cache == nullptr) {
		p_sum_cache = &dict::global_sum_cache;
	}
	// normalize as a whole
	auto renorm_res = sum_weights_normalize(w_node1.weight, w_node2.weight);
	weightednode next_wnode1 = weightednode{ renorm_res.nweight1, w_node1.p_node };
	weightednode next_wnode2 = weightednode{ renorm_res.nweight2, w_node2.p_node };
	auto res = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef, *p_sum_cache);
	return res;
}

weightednode wnode::contract_iterate(weightednode w_node, int dim_data, const int64_t* p_data_shape,
	int num_remained, const int* p_r_i1, const int* p_r_i2,
	int num_waiting, const int* p_w_i, const int* p_w_v, dict::sum_table& sum_cache, dict::cont_table& cont_cache) {

	auto p_node = w_node.p_node;
	if (p_node == TERMINAL_NODE) {
		// close all the unprocessed indices
		double scale = 1.;
		for (int i = 0; i < num_remained; i++) {
			scale *= p_data_shape[i];
		}
		weightednode res = weightednode{ w_node.weight * scale, TERMINAL_NODE };
		return res;
	}

	weightednode res;


	// first look up in the dictionary
	dict::cont_key key = dict::cont_key(p_node->get_id(), num_remained, p_r_i1, p_r_i2, num_waiting, p_w_i, p_w_v);
	auto p_find_res = cont_cache.find(key);
	if (p_find_res != cont_cache.end()) {
		res = p_find_res->second;
		res.weight = res.weight * w_node.weight;
		return res;
	}
	else {

		auto order = p_node->get_order();

		// store the scaling number due to skipped remained indices
		double scale = 1.;

		// process the skipped remained indices (situations only first index skipped will be processed afterwards)
		auto temp_size = num_remained > num_waiting ? num_remained : num_waiting;
		auto ptemp_ls_0 = (int*)malloc(sizeof(int) * temp_size);
		auto ptemp_ls_1 = (int*)malloc(sizeof(int) * temp_size);
		int num_remained_pd = 0;
		for (int i = 0; i < num_remained; i++) {
			if (p_r_i2[i] >= order) {
				ptemp_ls_0[num_remained_pd] = p_r_i1[i];
				ptemp_ls_1[num_remained_pd] = p_r_i2[i];
				num_remained_pd++;
			}
			else {
				scale *= p_data_shape[p_r_i1[i]];
			}
		}
		int* p_r_i1_pd = array_clone(ptemp_ls_0, num_remained_pd);
		int* p_r_i2_pd = array_clone(ptemp_ls_1, num_remained_pd);

		// process the skipped waiting indcies
		int num_waiting_pd = 0;
		for (int i = 0; i < num_waiting; i++) {
			// if a waiting index is skipped, remove it from the waiting index list
			if (p_w_i[i] >= order) {
				ptemp_ls_0[num_waiting_pd] = p_w_i[i];
				ptemp_ls_1[num_waiting_pd] = p_w_v[i];
				num_waiting_pd++;
			}
		}
		int* p_w_i_pd = array_clone(ptemp_ls_0, num_waiting_pd);
		int* p_w_v_pd = array_clone(ptemp_ls_1, num_waiting_pd);

		// free the temp memory
		free(ptemp_ls_0);
		free(ptemp_ls_1);

		// the flag for no operation performed
		bool not_operated = true;

		// check whether all operations have already taken place
		if (num_remained_pd == 0 && num_waiting_pd == 0) {
			res = weightednode{ wcomplex(1.,0.) * scale, p_node };
			not_operated = false;
		}
		else if (num_waiting_pd != 0) {
			/*
			*   waiting_ils is not empty in this case
			*	If multiple waiting indices have been skipped, we will resolve with iteration, one by one.
			*/
			auto next_to_close = min(p_w_i_pd, num_waiting_pd);

			// note that next_i_to_close >= node,order is guaranteed here
			if (order == next_to_close.second) {
				// close the waiting index
				int* next_p_w_i = removed(p_w_i_pd, num_waiting_pd, next_to_close.first);
				int* next_p_w_v = removed(p_w_v_pd, num_waiting_pd, next_to_close.first);
				weightednode next_wnode = weightednode{
					p_node->get_weights()[p_w_v_pd[next_to_close.first]],
					p_node->get_successors()[p_w_v_pd[next_to_close.first]]
				};
				res = contract_iterate(next_wnode, dim_data, p_data_shape,
					num_remained_pd, p_r_i1_pd, p_r_i2_pd,
					num_waiting_pd - 1, next_p_w_i, next_p_w_v, sum_cache, cont_cache);
				free(next_p_w_i);
				free(next_p_w_v);
				res.weight = res.weight * scale;
				not_operated = false;
			}


		}
		if (num_remained_pd != 0 && not_operated) {
			/*
			*	Check the remained indices to start tracing.
			*	If multiple (smaller ones of) remained indices have been skipped, we will resolve with iteration, one by one.
			*/
			auto next_to_open = min(p_r_i1_pd, num_remained_pd);
			if (order >= next_to_open.second) {
				// open the index and finally sum up

				// next_r_ils: already sorted.
				int* next_p_r_i1 = removed(p_r_i1_pd, num_remained_pd, next_to_open.first);
				int* next_p_r_i2 = removed(p_r_i2_pd, num_remained_pd, next_to_open.first);

				// find the right insert place in waiting_ils
				int insert_pos = 0;
				for (int i = 0; i < num_waiting_pd; i++) {
					if (p_w_i_pd[i] < next_to_open.second) {
						insert_pos += 1;
					}
					else {
						break;
					}
				}

				//get the index range
				node_int range;
				if (order == next_to_open.second) {
					range = p_node->get_range();
				}
				else {
					range = p_data_shape[next_to_open.second];
				}

				weightednode* p_wnodes = (weightednode*)malloc(sizeof(weightednode) * range);
				auto this_weights = p_node->get_weights();
				auto this_successors = p_node->get_successors();

				int* next_p_w_i = inserted(p_w_i_pd, num_waiting_pd, insert_pos, 0);
				int* next_p_w_v = inserted(p_w_v_pd, num_waiting_pd, insert_pos, 0);
				if (order == next_to_open.second) {
					for (int i = 0; i < range; i++) {
						weightednode new_wnode;
						if (this_successors[i] == TERMINAL_NODE) {
							new_wnode.p_node = TERMINAL_NODE;
							new_wnode.weight = this_weights[i];
						}
						else {
							// produce the sorted new index lists
							next_p_w_i[insert_pos] = p_r_i2_pd[next_to_open.first];
							next_p_w_v[insert_pos] = i;
							weightednode next_wnode = weightednode{
								this_weights[i],
								this_successors[i]
							};
							new_wnode = contract_iterate(next_wnode, dim_data, p_data_shape,
								num_remained_pd - 1, next_p_r_i1, next_p_r_i2,
								num_waiting_pd + 1, next_p_w_i, next_p_w_v, sum_cache, cont_cache);
						}
						p_wnodes[i] = new_wnode;
					}
				}
				else {
					// this node skipped the index next_i_to_open in this case
					for (int i = 0; i < range; i++) {
						// produce the sorted new index lists
						next_p_w_i[insert_pos] = p_r_i2_pd[next_to_open.first];
						next_p_w_v[insert_pos] = i;
						p_wnodes[i] = contract_iterate(w_node, dim_data, p_data_shape,
							num_remained_pd - 1, next_p_r_i1, next_p_r_i2,
							num_waiting_pd + 1, next_p_w_i, next_p_w_v, sum_cache, cont_cache);
					}
				}
				free(next_p_w_i);
				free(next_p_w_v);
				free(next_p_r_i1);
				free(next_p_r_i2);

				// however the subnode outcomes are calculated, sum them over.
				res = p_wnodes[0];
				for (int i = 1; i < range; i++) {
					res = sum(res, p_wnodes[i], &sum_cache);
				}
				free(p_wnodes);
				res.weight = res.weight * scale;
				not_operated = false;
			}
		}

		if (not_operated) {
			// in this case, no operation can be performed on this node, so we move on the the following nodes.
			auto range = p_node->get_range();
			wcomplex* p_weights = (wcomplex*)malloc(sizeof(wcomplex) * range);
			const Node** p_nodes = (const Node**)malloc(sizeof(const Node*) * range);
			auto this_weights = p_node->get_weights();
			auto this_successors = p_node->get_successors();
			for (int i = 0; i < range; i++) {
				weightednode new_wnode;
				if (this_successors[i] == TERMINAL_NODE) {
					new_wnode.p_node = TERMINAL_NODE;
					new_wnode.weight = this_weights[i];
				}
				else {
					weightednode next_wnode = weightednode{
						this_weights[i],
						this_successors[i]
					};
					new_wnode = contract_iterate(next_wnode, dim_data, p_data_shape,
						num_remained_pd, p_r_i1_pd, p_r_i2_pd,
						num_waiting_pd, p_w_i_pd, p_w_v_pd, sum_cache, cont_cache);
				}
				p_weights[i] = new_wnode.weight;
				p_nodes[i] = new_wnode.p_node;
			}
			auto&& temp_new_node = Node(0, order, range, p_weights, p_nodes);
			res = normalize(weightednode{ wcomplex(1.,0.),&temp_new_node });
		}

		free(p_r_i1_pd);
		free(p_r_i2_pd);
		free(p_w_i_pd);
		free(p_w_v_pd);
		cont_cache.insert(make_pair(key, res));
		res.weight = res.weight * w_node.weight;
		return res;
	}
}

weightednode wnode::contract(weightednode w_node, int dim_data, const int64_t* p_data_shape,
	int num_pair, const int* p_i1, const int* p_i2) {


	auto&& res = contract_iterate(w_node, dim_data, p_data_shape, num_pair, p_i1, p_i2, 0, nullptr, nullptr,
		dict::global_sum_cache, dict::global_cont_cache);

	// shift the nodes at a time

	/////////////////////////////////
	//prepare the new index order
	int* p_new_order = (int*)malloc(sizeof(int) * dim_data);
	auto p_reduced_indices = array_concat(p_i1, num_pair, p_i2, num_pair);
	sort(p_reduced_indices, p_reduced_indices + num_pair * 2);
	for (int i = 0; i < p_reduced_indices[0]; i++) {
		p_new_order[i] = i;
	}
	for (int i = 0; i < num_pair * 2 - 1; i++) {
		for (int j = p_reduced_indices[i] + 1; j < p_reduced_indices[i + 1]; j++) {
			p_new_order[j] = j - i - 1;
		}
	}
	for (int j = p_reduced_indices[num_pair * 2 - 1] + 1; j < dim_data; j++) {
		p_new_order[j] = j - 2 * num_pair;
	}
	/////////////////////////////////
	res.p_node = Node::shift_multiple(res.p_node, dim_data, p_new_order);
	free(p_reduced_indices);
	free(p_new_order);
	return res;
}
