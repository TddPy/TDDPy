#pragma once
#include "node.hpp"


template <class W>
class wnode {
public:
	inline static bool is_equal(const node::weightednode<W>& a, const node::weightednode<W>& b) {
		return (a.node == b.node && weight::is_equal(a.weight, b.weight));
	}

	/// <summary>
	/// To create the weighted node iteratively according to the instructions.
	/// </summary>
	/// <returns></returns>
	static node::weightednode<W> as_tensor_iterate(const CUDAcpl::Tensor& t,
		const std::vector<int64_t>& parallel_shape,
		const std::vector<int64_t>& data_shape,
		const std::vector<int64_t>& index_order, int depth) {

		node::weightednode<W> res;
		// checks whether the tensor is reduced to the [[...[val]...]] form
		auto dim_data = data_shape.size() - 1;
		if (depth == dim_data) {
			if (dim_data == 0) {
				res.weight = CUDAcpl::item(t);
			}
			else {
				res.weight = CUDAcpl::item(t);
			}
			res.node = nullptr;
			return res;
		}


		int split_pos = index_order[depth];
		auto split_tensor = t.split(1, t.dim() - dim_data + split_pos - 1);
		// -1 is because the extra inner dim for real and imag

		auto new_successors = std::vector<node::weightednode<W>>(data_shape[split_pos]);
		for (int i = 0; i < data_shape[split_pos]; i++) {
			new_successors[i] = as_tensor_iterate(split_tensor[i], parallel_shape, data_shape, index_order, depth + 1);
		}
		auto&& temp_node = node::Node<W>(depth, std::move(new_successors));
		// normalize this depth
		return normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
	}

	/// <summary>
	/// Conduct the normalization of this wnode.
	/// This method only normalize the given wnode, and assumes the wnodes under it are already normalized.
	/// </summary>
	/// <param name="w_node"></param>
	/// <returns>Return the normalized node and normalization coefficients as a wnode.</returns>
	static node::weightednode<W> normalize(const node::weightednode<W>& w_node) {
		if (w_node.node == nullptr) {
			return node::weightednode<W>(w_node);
		}

		// redirect zero weighted nodes to the terminal node
		if (weight::is_zero(w_node.weight)) {
			return node::weightednode<W>(wcomplex(0., 0.), nullptr);
		}

		auto&& successors = w_node.node->get_successors();

		// node reduction check (reduce when all equal)
		bool all_equal = true;
		for (auto i = successors.begin() + 1; i != successors.end(); i++) {
			if (!is_equal(successors[0], *i)) {
				all_equal = false;
				break;
			}
		}
		if (all_equal) {
			return node::weightednode<W>(
				w_node.weight * successors[0].weight,
				successors[0].node
				);
		}

		// check whether all successor weights are zero, and redirect to terminal node if so
		bool all_zero = true;
		for (auto i = successors.begin(); i != successors.end(); i++) {
			if (!weight::is_zero(i->weight)) {
				all_zero = false;
				break;
			}
		}
		if (all_zero) {
			return node::weightednode<W>(wcomplex(0., 0.), nullptr);
		}


		// start to normalize the weights
		int i_max = 0;
		double norm_max = norm(successors[0].weight);
		for (int i = 1; i < successors.size(); i++) {
			double temp_norm = norm(successors[i].weight);
			if (temp_norm > norm_max) {
				i_max = i;
				norm_max = temp_norm;
			}
		}
		wcomplex weig_max = successors[i_max].weight;
		auto&& new_successors = std::vector<node::weightednode<W>>(successors);
		for (auto i = new_successors.begin(); i != new_successors.end(); i++) {
			i->weight = i->weight / weig_max;
		}
		auto new_node = node::Node<W>::get_unique_node(w_node.node->get_order(), new_successors);
		return node::weightednode<W>{ weig_max* w_node.weight, new_node };
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="w_node">Note that w_node.node should not be nullptr</param>
	/// <returns>tensor of dim (dim_data - node.order + 1)</returns>
	static CUDAcpl::Tensor to_CUDAcpl_iterate(const node::weightednode<W>& w_node, const std::vector<int64_t>& data_shape) {
		// w_node.node is guaranteed not to be null

		auto current_order = w_node.node->get_order();

		auto par_tensor = std::vector<CUDAcpl::Tensor>(w_node.node->get_range());
		auto&& successors = w_node.node->get_successors();

		auto dim_data = data_shape.size() - 1;

		// The temp shape for adjustment.
		auto new_len = dim_data - current_order;
		auto p_temp_shape = std::vector<int64_t>(new_len);
		p_temp_shape[new_len - 1] = 2;


		CUDAcpl::Tensor temp_tensor;
		CUDAcpl::Tensor uniform_tensor;
		int next_order = 0;
		auto i_par = par_tensor.begin();
		for (auto i = successors.begin(); i != successors.end(); i++) {
			// detect terminal nodes, or iterate on the next node
			if (i->node == nullptr) {
				temp_tensor = std::move(CUDAcpl::from_complex(i->weight));
				next_order = dim_data;
			}
			else {
				// first look up in the dictionary
				auto key = i->node->get_id();
				auto p_find_res = cache::Global_Cache<W>::p_CUDAcpl_cache->find(key);
				if (p_find_res != cache::Global_Cache<W>::p_CUDAcpl_cache->end()) {
					uniform_tensor = p_find_res->second;
				}
				else {
					auto&& next_wnode = node::weightednode<W>(wcomplex(1., 0.), i->node);
					uniform_tensor = to_CUDAcpl_iterate(next_wnode, data_shape);
					// add into the dictionary
					cache::Global_Cache<W>::p_CUDAcpl_cache->insert(std::make_pair(std::move(key), uniform_tensor));
				}
				next_order = i->node->get_order();
				// multiply the dangling weight
				temp_tensor = CUDAcpl::mul_element_wise(uniform_tensor, i->weight);
			}

			// broadcast according to the index distance
			if (next_order - current_order > 1) {
				// prepare the new data shape
				for (int i = 0; i < next_order - current_order - 1; i++) {
					p_temp_shape[i] = 1;
				}
				for (int i = 0; i < temp_tensor.dim() - 1; i++) {
					p_temp_shape[i + next_order - current_order - 1] = temp_tensor.size(i);
				}
				temp_tensor = temp_tensor.view(c10::IntArrayRef(p_temp_shape));
				for (int i = 0; i < next_order - current_order - 1; i++) {
					p_temp_shape[i] = data_shape[i + current_order + 1];
				}
				temp_tensor = temp_tensor.expand(c10::IntArrayRef(p_temp_shape));
			}
			*i_par = std::move(temp_tensor);
			i_par++;
		}
		auto&& res = torch::stack(par_tensor, 0);
		// multiply the dangling weight and return
		return CUDAcpl::mul_element_wise(res, w_node.weight);
	}

	/// <summary>
	/// Get the CUDAcpl_Tensor determined from this node and the weights.
	///(use the trival index order)
	/// </summary>
	/// <param name="w_node"></param>
	/// <param name="inner_data_shape">data_shape(in the corresponding inner index order) is required, for the result should broadcast at reduced nodes of indices.
	/// Note that an *extra dimension* of 2 is needed at the end of p_inner_data_shape.</param>
	/// <returns></returns>
	static CUDAcpl::Tensor to_CUDAcpl(const node::weightednode<W>& w_node, const std::vector<int64_t>& inner_data_shape) {
		int n_extra_one = 0;
		auto dim_data = inner_data_shape.size() - 1;
		CUDAcpl::Tensor res;
		if (w_node.node == nullptr) {
			res = CUDAcpl::mul_element_wise(CUDAcpl::ones({}), w_node.weight);
			n_extra_one = dim_data;
		}
		else {
			res = to_CUDAcpl_iterate(w_node, inner_data_shape);
			n_extra_one = w_node.node->get_order();
		}
		// this extra layer is for adding the reduced dimensions at the front
		// prepare the real data shape
		auto full_data_shape = std::vector<int64_t>(dim_data + 1);
		full_data_shape[dim_data] = 2;
		// res.dim() == dim_data + 1 - n_extra_one should hold
		auto&& cur_data_shape = res.sizes();
		for (int i = 0; i < n_extra_one; i++) {
			full_data_shape[i] = 1;
		}
		for (int i = 0; i < cur_data_shape.size(); i++) {
			full_data_shape[i + n_extra_one] = cur_data_shape[i];
		}
		res = res.view(c10::IntArrayRef(full_data_shape));
		res = res.expand(c10::IntArrayRef(inner_data_shape));
		return res;
	}


	static node::weightednode<W> direct_product(const node::weightednode<W>& a, int a_depth,
		const node::weightednode<W>& b, bool parallel_tensor) {
		wcomplex weight;
		if (parallel_tensor) {
			// not implemented yet
			throw - 10;
		}
		else {
			weight = a.weight * b.weight;
		}
		auto p_res_node = node::Node<W>::append(a.node, a_depth, b.node, parallel_tensor);
		return node::weightednode<W>(std::move(weight), p_res_node);
	}


	/// <summary>
	/// Process the weights, and normalize them as a whole. Return (new_weights1, new_weights2, renorm_coef).
	///	The strategy to produce the weights(for every individual element) :
	///		maximum_norm: = max(weight1.norm, weight2.norm)
	///		1. if maximum_norm > EPS: renormalize max_weight to 1.
	///		3. else key is 0000..., 0000...
	///	Renormalization coefficient for zero items are left to be 1.
	/// </summary>
	/// <param name="weight1"></param>
	/// <param name="weight2"></param>
	/// <returns></returns>
	inline static weight::sum_nweights<W> sum_weights_normalize(const W& weight1, const W& weight2) {
		wcomplex renorm_coef = (norm(weight1) > norm(weight2)) ? weight1 : weight2;

		auto nweight1 = wcomplex(0., 0.);
		auto nweight2 = wcomplex(0., 0.);
		if (norm(renorm_coef) > weight::EPS) {
			nweight1 = weight1 / renorm_coef;
			nweight2 = weight2 / renorm_coef;
		}
		else {
			renorm_coef = wcomplex(1., 0.);
		}
		return weight::sum_nweights<W>(
			std::move(nweight1),
			std::move(nweight2),
			std::move(renorm_coef)
			);
	}


	/// <summary>
	/// Sum up the given weighted nodes, multiply the renorm_coef, and return the reduced weighted node result.
	/// Note that weights1 and weights2 as a whole should have been normalized,
	/// and renorm_coef is the coefficient.
	/// </summary>
	/// <param name="w_node1"></param>
	/// <param name="w_node2"></param>
	/// <param name="renorm_coef"></param>
	/// <param name="sum_cache"></param>
	/// <returns></returns>
	static node::weightednode<W> sum_iterate(const node::weightednode<W>& w_node1,
		const node::weightednode<W>& w_node2, const W& renorm_coef) {
		if (w_node1.node == nullptr && w_node2.node == nullptr) {
			return node::weightednode<W>((w_node1.weight + w_node2.weight) * renorm_coef, nullptr);
		}

		// produce the unique key and look up in the cache
		auto key = cache::sum_key<W>(
			node::Node<W>::get_id_all(w_node1.node), w_node1.weight,
			node::Node<W>::get_id_all(w_node2.node), w_node2.weight);

		auto p_find_res = cache::Global_Cache<W>::p_sum_cache->find(key);
		if (p_find_res != cache::Global_Cache<W>::p_sum_cache->end()) {
			node::weightednode<W> res = p_find_res->second;
			res.weight = res.weight * renorm_coef;
			return res;
		}
		else {
			///////////////////////////////////////////////////////////////////////
			// ensure w_node1.node to be the node of smaller order, through swaping
			///////////////////////////////////////////////////////////////////////
			const node::weightednode<W>* p_wnode_1, * p_wnode_2;
			if (w_node1.node == nullptr ||
				w_node1.node->get_order() > w_node2.node->get_order()) {
				p_wnode_1 = &w_node2;
				p_wnode_2 = &w_node1;
			}
			else {
				p_wnode_1 = &w_node1;
				p_wnode_2 = &w_node2;
			}

			auto new_successors = std::vector<node::weightednode<W>>(p_wnode_1->node->get_range());
			if (p_wnode_2->node != nullptr &&
				p_wnode_1->node->get_order() == p_wnode_2->node->get_order()) {
				// node1 and node2 are assumed to have the same shape
				auto&& successors_1 = p_wnode_1->node->get_successors();
				auto&& successors_2 = p_wnode_2->node->get_successors();
				auto i_1 = successors_1.begin();
				auto i_2 = successors_2.begin();
				auto i_new = new_successors.begin();
				for (; i_1 != successors_1.end(); i_1++, i_2++, i_new++) {
					// normalize as a whole
					auto&& renorm_res = sum_weights_normalize(p_wnode_1->weight * i_1->weight, p_wnode_2->weight * i_2->weight);
					auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->node);
					auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), i_2->node);
					*i_new = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
				}
			}
			else {
				/*
					There are three cases following, corresponding to the same procedure:
					1. node1 == None, node2 != None
					2. node2 == None, node1 != None
					3. node1 != None, node2 != None, but node1.order != noder2.order
					We have analysised the situation to reuse the codes.
					wnode_1 will be the lower ordered weighted node.
				*/

				auto&& successors = p_wnode_1->node->get_successors();
				auto i_1 = successors.begin();
				auto i_new = new_successors.begin();
				for (; i_1 != successors.end(); i_1++, i_new++) {
					auto&& renorm_res = sum_weights_normalize(p_wnode_1->weight * i_1->weight, p_wnode_2->weight);
					auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->node);
					auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), p_wnode_2->node);
					*i_new = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
				}
			}

			auto&& temp_node = node::Node<W>(p_wnode_1->node->get_order(), std::move(new_successors));
			auto&& res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));

			// cache the result
			cache::Global_Cache<W>::p_sum_cache->insert(std::make_pair(std::move(key), res));

			// multiply the renorm_coef and return
			res.weight = res.weight * renorm_coef;
			return res;
		}
	}

	/// <summary>
	/// sum two weighted nodes.
	/// </summary>
	/// <param name="w_node1"></param>
	/// <param name="w_node2"></param>
	/// <returns></returns>
	static node::weightednode<W> sum(const node::weightednode<W> w_node1, const node::weightednode<W> w_node2) {
		// normalize as a whole
		auto&& renorm_res = sum_weights_normalize(w_node1.weight, w_node2.weight);
		auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), w_node1.node);
		auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), w_node2.node);
		return sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
	}
};