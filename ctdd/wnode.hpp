#pragma once
#include "node.hpp"


template <class W>
class wnode {
private:
	inline static bool cmd_first_smaller(const std::pair<int, int>& a, const std::pair<int, int>& b) {
		return a.first < b.first;
	}
	inline static bool cmd_second_smaller(const std::pair<int, int>& a, const std::pair<int, int>& b) {
		return a.second < b.second;
	}
	inline static int get_cmd_insert_pos(const cache::pair_cmd& cmd_ls, int order) {
		int insert_pos = 0;
		for (; insert_pos < cmd_ls.size(); insert_pos++) {
			if (cmd_ls[insert_pos].first < order) {
				continue;
			}
			else {
				break;
			}
		}
		return insert_pos;
	}



public:
	/// <summary>
	/// Get the new order, when reduced indices are removed, and the left indices in order.
	/// </summary>
	/// <param name="length"></param>
	/// <param name="reduced_indices">must be in order</param>
	/// <returns></returns>
	inline static std::vector<int64_t> get_new_order(int length, const std::vector<int64_t>& reduced_indices) {
		////////////////////////////////
		// prepare the new index order
		if (reduced_indices.empty()) {
			std::vector<int64_t> new_order(length);
			for (int i = 0; i < length; i++) {
				new_order[i] = i;
			}
			return new_order;
		}

		auto&& new_order = std::vector<int64_t>(length);
		for (int i = 0; i < reduced_indices[0]; i++) {
			new_order[i] = i;
		}
		for (int i = 0; i < reduced_indices.size() - 1; i++) {
			for (int j = reduced_indices[i] + 1; j < reduced_indices[i + 1]; j++) {
				new_order[j] = j - i - 1;
			}
		}
		for (int j = reduced_indices[reduced_indices.size() - 1] + 1; j < length; j++) {
			new_order[j] = j - reduced_indices.size();
		}
		////////////////////////////////
		return new_order;
	}

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
		auto&& dim_data = data_shape.size() - 1;
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
		int axe_pos = t.dim() - dim_data + split_pos - 1;

		//note: torch::chunk does not work here

		auto&& new_successors = std::vector<node::weightednode<W>>(data_shape[split_pos]);
		for (int i = 0; i < data_shape[split_pos]; i++) {
			new_successors[i] = as_tensor_iterate(
				// -1 is because the extra inner dim for real and imag
				t.select(axe_pos, i).unsqueeze(axe_pos),
				parallel_shape, data_shape, index_order, depth + 1);
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
		for (const auto& succ : successors) {
			if (!is_equal(successors[0], succ)) {
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
		for (const auto& succ : successors) {
			if (!weight::is_zero(succ.weight)) {
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
		for (auto&& succ : new_successors) {
			succ.weight = succ.weight / weig_max;
		}
		auto&& new_node = node::Node<W>::get_unique_node(w_node.node->get_order(), new_successors);
		return node::weightednode<W>{ weig_max* w_node.weight, new_node };
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="w_node">Note that w_node.node should not be nullptr</param>
	/// <returns>tensor of dim (dim_data - node.order + 1)</returns>
	static CUDAcpl::Tensor to_CUDAcpl_iterate(const node::weightednode<W>& w_node, const std::vector<int64_t>& data_shape) {
		// w_node.node is guaranteed not to be null

		auto&& current_order = w_node.node->get_order();

		auto&& par_tensor = std::vector<CUDAcpl::Tensor>(w_node.node->get_range());
		auto&& successors = w_node.node->get_successors();

		auto&& dim_data = data_shape.size() - 1;

		// The temp shape for adjustment.(shape of the successors of this node)
		auto&& p_temp_shape = std::vector<int64_t>(dim_data - current_order);
		p_temp_shape[dim_data - current_order - 1] = 2;


		CUDAcpl::Tensor temp_tensor;
		CUDAcpl::Tensor uniform_tensor;
		int next_order = 0;
		auto&& i_par = par_tensor.begin();
		for (auto&& i = successors.cbegin(); i != successors.cend(); i++, i_par++) {
			// detect terminal nodes, or iterate on the next node
			if (i->node == nullptr) {
				temp_tensor = CUDAcpl::from_complex(i->weight);
				next_order = dim_data;
			}
			else {
				// first look up in the dictionary
				auto&& key = cache::CUDAcpl_table_key<W>(i->node->get_id(), data_shape);
				auto&& p_find_res = cache::Global_Cache<W>::p_CUDAcpl_cache->find(key);
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
		auto&& dim_data = inner_data_shape.size() - 1;
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
		auto&& full_data_shape = std::vector<int64_t>(dim_data + 1);
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
		const node::weightednode<W>& b,
		const std::vector<int64_t>& parallel_shape,
		const std::vector<int64_t>& shape_front,
		const std::vector<int64_t>& shape_back,
		bool parallel_tensor) {
		wcomplex weight;
		if (parallel_tensor) {
			// not implemented yet
			throw - 10;
		}
		else {
			weight = a.weight * b.weight;
		}
		auto&& p_res_node = node::Node<W>::append(a.node, a_depth, b.node, parallel_shape, shape_front, shape_back, parallel_tensor);
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
	inline static weight::sum_nweights<W> weights_normalize(const W& weight1, const W& weight2) {
		wcomplex renorm_coef = (norm(weight1) > norm(weight2)) ? weight1 : weight2;

		auto&& nweight1 = wcomplex(0., 0.);
		auto&& nweight2 = wcomplex(0., 0.);
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
		auto&& key = cache::sum_key<W>(
			node::Node<W>::get_id_all(w_node1.node), w_node1.weight,
			node::Node<W>::get_id_all(w_node2.node), w_node2.weight);

		auto&& p_find_res = cache::Global_Cache<W>::p_sum_cache->find(key);
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
			if (w_node1.node == nullptr) {
				p_wnode_1 = &w_node2;
				p_wnode_2 = &w_node1;
			}
			else if (w_node2.node == nullptr) {
				p_wnode_1 = &w_node1;
				p_wnode_2 = &w_node2;
			}
			else if (w_node1.node->get_order() > w_node2.node->get_order()) {
				p_wnode_1 = &w_node2;
				p_wnode_2 = &w_node1;
			}
			else {
				p_wnode_1 = &w_node1;
				p_wnode_2 = &w_node2;
			}

			auto&& new_successors = std::vector<node::weightednode<W>>(p_wnode_1->node->get_range());

			bool not_operated = true;
			node::weightednode<W> res;
			if (p_wnode_2->node != nullptr) {

				// if they are the same node, then we can only adjust the weight
				if (p_wnode_1->node->get_id() == p_wnode_2->node->get_id()) {
					res = node::weightednode<W>(p_wnode_1->weight + p_wnode_2->weight, p_wnode_1->node);
					not_operated = false;
				}
				else if (p_wnode_1->node->get_order() == p_wnode_2->node->get_order()) {
					// node1 and node2 are assumed to have the same shape
					auto&& successors_1 = p_wnode_1->node->get_successors();
					auto&& successors_2 = p_wnode_2->node->get_successors();
					auto&& i_1 = successors_1.begin();
					auto&& i_2 = successors_2.begin();
					auto&& i_new = new_successors.begin();
					for (; i_1 != successors_1.end(); i_1++, i_2++, i_new++) {
						// normalize as a whole
						auto&& renorm_res = weights_normalize(p_wnode_1->weight * i_1->weight, p_wnode_2->weight * i_2->weight);
						auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->node);
						auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), i_2->node);
						*i_new = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
					}
					auto&& temp_node = node::Node<W>(p_wnode_1->node->get_order(), std::move(new_successors));
					res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
					not_operated = false;
				}
			}
			if (not_operated) {
				/*
					There are three cases following, corresponding to the same procedure:
					1. node1 == None, node2 != None
					2. node2 == None, node1 != None
					3. node1 != None, node2 != None, but node1.order != noder2.order
					We have analysised the situation to reuse the codes.
					wnode_1 will be the lower ordered weighted node.
				*/

				auto&& successors = p_wnode_1->node->get_successors();
				auto&& i_1 = successors.begin();
				auto&& i_new = new_successors.begin();
				for (; i_1 != successors.end(); i_1++, i_new++) {
					auto&& renorm_res = weights_normalize(p_wnode_1->weight * i_1->weight, p_wnode_2->weight);
					auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->node);
					auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), p_wnode_2->node);
					*i_new = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
				}
				auto&& temp_node = node::Node<W>(p_wnode_1->node->get_order(), std::move(new_successors));
				res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
			}


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
	static node::weightednode<W> sum(const node::weightednode<W>& w_node1, const node::weightednode<W>& w_node2) {
		// normalize as a whole
		auto&& renorm_res = weights_normalize(w_node1.weight, w_node2.weight);
		auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), w_node1.node);
		auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), w_node2.node);
		return sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
	}

	/// <summary>
	///	Note that :
	///	1. For remained_ls, we require smaller indices to be in the first place,
	///	and the corresponding larger one in the second place.
	///	2. indices in waiting_ls must be sorted in the asscending order to keep the cache key unique.
	///	3. The returning weighted nodes are NOT shifted. (node.order not adjusted)
	/// </summary>
	/// <returns></returns>
	static node::weightednode<W> trace_iterate(const node::weightednode<W>& w_node, const std::vector<int64_t>& data_shape,
		const cache::pair_cmd& remained_ls, const cache::pair_cmd& waiting_ls, const std::vector<int64_t>& new_order) {

		if (w_node.node == nullptr) {
			// close all the unprocessed indices
			double scale = 1.;
			for (const auto& cmd : remained_ls) {
				scale *= data_shape[cmd.first];
			}
			return node::weightednode<W>(w_node.weight * scale, nullptr);
		}

		node::weightednode<W> res;

		// first look up in the dictionary
		auto&& key = cache::trace_key<W>(w_node.node->get_id(), remained_ls, waiting_ls);
		auto&& p_find_res = cache::Global_Cache<W>::p_trace_cache->find(key);
		if (p_find_res != cache::Global_Cache<W>::p_trace_cache->end()) {
			res = p_find_res->second;
			res.weight = res.weight * w_node.weight;
			return res;
		}
		else {
			auto&& order = w_node.node->get_order();

			// store the scaling number due to skipped remained indices
			double scale = 1.;

			// process the skipped remained indices (situations only first index skipped will be processed afterwards)
			auto&& remained_ls_pd = cache::pair_cmd();
			for (const auto& cmd : remained_ls) {
				if (cmd.second >= order) {
					remained_ls_pd.push_back(cmd);
				}
				else {
					scale *= data_shape[cmd.first];
				}
			}
			auto&& waiting_ls_pd = cache::pair_cmd();
			for (const auto& cmd : waiting_ls) {
				if (cmd.first >= order) {
					waiting_ls_pd.push_back(cmd);
				}
			}

			// the flag for no operation performed
			bool not_operated = true;

			auto&& successors = w_node.node->get_successors();

			// check whether all operations have already taken place
			if (remained_ls_pd.empty() && waiting_ls_pd.empty()) {
				res = node::weightednode<W>(wcomplex(1., 0.), w_node.node);
				not_operated = false;
			}
			else if (!waiting_ls_pd.empty()) {
				/*
				*   waiting_ils is not empty in this case
				*	If multiple waiting indices have been skipped, we will resolve with iteration, one by one.
				*/

				// next_to_close: <pos in waiting_ls_pd, min in waiting_ls_pd>: must be the first of waiting_ls_pd
				// note that next_i_to_close >= node.order is guaranteed here
				if (order == waiting_ls_pd[0].first) {
					auto&& index_val = waiting_ls_pd[0].second;
					// close the waiting index
					auto&& next_waiting_ls = removed<std::pair<int, int>>(waiting_ls_pd, 0);
					res = trace_iterate(successors[index_val], data_shape, remained_ls_pd, next_waiting_ls, new_order);
					not_operated = false;
				}
			}
			if (!remained_ls_pd.empty() && not_operated) {
				/*
				*	Check the remained indices to start tracing.
				*	If multiple (smaller ones of) remained indices have been skipped, we will resolve with iteration, one by one.
				*/
				if (order >= remained_ls_pd[0].first) {
					// open the index and finally sum up

					auto&& next_remained_ls = removed(remained_ls_pd, 0);

					// find the right insert place in waiting_ls
					int insert_pos = get_cmd_insert_pos(waiting_ls_pd, remained_ls_pd[0].second);

					// get the index range
					int range = data_shape[remained_ls_pd[0].first];

					// produce the sorted new index lists
					auto&& next_waiting_ls = inserted(waiting_ls_pd, insert_pos, std::make_pair(
						remained_ls_pd[0].second, 0)
					);

					auto&& new_successors = std::vector<node::weightednode<W>>(range);
					if (order == remained_ls_pd[0].first) {
						int index_val = 0;
						auto&& i_new = new_successors.begin();
						for (auto&& i = successors.begin(); i != successors.end(); i++, i_new++, index_val++) {
							if (i->node == nullptr) {
								i_new->node = nullptr;
								i_new->weight = i->weight;
							}
							else {
								// adjust the new index value
								next_waiting_ls[insert_pos].second = index_val;
								*i_new = trace_iterate(*i, data_shape, next_remained_ls, next_waiting_ls, new_order);
							}
						}
					}
					else {
						// this node skipped the index next_i_to_open in this case
						int index_val = 0;
						for (auto&& succ_new : new_successors) {
							next_waiting_ls[insert_pos].second = index_val;
							succ_new = trace_iterate(node::weightednode<W>(wcomplex(1., 0.), w_node.node),
								data_shape, next_remained_ls, next_waiting_ls, new_order);
							index_val++;
						}
					}

					// however the subnode outcomes are calculated, sum them over.
					res = new_successors[0];
					for (auto&& i = new_successors.begin() + 1; i != new_successors.end(); i++) {
						res = sum(res, *i);
					}
					not_operated = false;
				}
			}

			if (not_operated) {
				// in this case, no operation can be performed on this node, so we move on the the following nodes.
				auto&& new_successors = std::vector<node::weightednode<W>>(w_node.node->get_range());
				auto&& i_new = new_successors.begin();
				for (auto&& i = successors.begin(); i != successors.end(); i++, i_new++) {
					if (i->node == nullptr) {
						i_new->node = nullptr;
						i_new->weight = i->weight;
					}
					else {
						*i_new = trace_iterate(*i, data_shape, remained_ls_pd, waiting_ls_pd, new_order);
					}
				}
				auto&& temp_node = node::Node<W>(order, std::move(new_successors));
				res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
			}

			// add to the cache
			res.weight *= scale;
			cache::Global_Cache<W>::p_trace_cache->insert(std::make_pair(std::move(key), res));
			res.weight = res.weight * w_node.weight;
			return res;
		}
	}


	/// <summary>
	/// Trace the weighted node according to the specified data_indices. Return the reduced result.
	///	e.g. ([a, b, c], [d, e, f]) means tracing indices a - d, b - e, c - f(of course two lists should be in the same size)
	/// </summary>
	/// <param name="w_node"></param>
	/// <param name="dim_data"></param>
	/// <param name="data_shape">correponds to data_indices</param>
	/// <param name="remained_ls">data_indices should be counted in the data indices only.(smaller indices are required to be in the first place.)</param>
	/// <param name="reduced_indices"> must be sorted </param>
	/// <returns></returns>
	static node::weightednode<W> trace(const node::weightednode<W>& w_node, const std::vector<int64_t>& data_shape,
		const cache::pair_cmd& remained_ls, const std::vector<int64_t> reduced_indices) {

		// sort the remained_ls by first element, to keep the key unique
		cache::pair_cmd sorted_remained_ls(remained_ls);

		std::sort(sorted_remained_ls.begin(), sorted_remained_ls.end(),
			[](const std::pair<int,int>& a, const std::pair<int,int>& b) {
				return (a.first < b.first);
			});

		auto&& num_pair = remained_ls.size();

		////////////////////////////////
		// prepare the new index order
		auto&& new_order = get_new_order(data_shape.size() - 1, reduced_indices);
		////////////////////////////////

		auto&& res = trace_iterate(w_node, data_shape, sorted_remained_ls, cache::pair_cmd(), new_order);

		res.node = node::Node<W>::shift_multiple(res.node, new_order);

		return res;
	}

	/// <summary>
	/// Contract the two weighted_nodes A and B.
	/// </summary>
	/// <param name="p_node_a"></param>
	/// <param name="p_node_b"></param>
	/// <param name="weight"></param>
	/// <param name="data_shape_a"></param>
	/// <param name="data_shape_b"></param>
	/// <param name="remained_ls">should be sorted by the first element.</param>
	/// <param name="a_waiting_ls">should be sorted by the first element.</param>
	/// <param name="b_waiting_ls">should be sorted by the first element.</param>
	/// <param name="a_new_order">the new order of each node in A</param>
	/// <param name="b_new_order">the new order of each node in B</param>
	/// <param name="parallel_tensor">whether to tensor on the parallel indices</param>
	/// <returns></returns>
	static node::weightednode<W> contract_iterate(
		const node::Node<W>* p_node_a, const node::Node<W>* p_node_b, const W& weight,
		const std::vector<int64_t>& data_shape_a, const std::vector<int64_t>& data_shape_b,
		const cache::pair_cmd& remained_ls,
		const cache::pair_cmd& a_waiting_ls, const cache::pair_cmd& b_waiting_ls,
		const std::vector<int64_t>& a_new_order, const std::vector<int64_t>& b_new_order, bool parallel_tensor) {
		if (p_node_a == nullptr && p_node_b == nullptr) {
			// close all the unprocessed indices
			double scale = 1.;
			for (const auto& cmd : remained_ls) {
				scale *= data_shape_a[cmd.first];
			}
			return node::weightednode<W>(weight * scale, nullptr);
		}

		node::weightednode<W> res;

		auto&& order_a = p_node_a == nullptr ? data_shape_a.size() - 1 : p_node_a->get_order();
		auto&& order_b = p_node_b == nullptr ? data_shape_b.size() - 1 : p_node_b->get_order();
		// first look up in the dictionary
		// note that the results stored in cache is calculated with the uniformed (1., w_node_a.node) and (1., w_node_b.node)
		auto&& key = cache::cont_key<W>(p_node_a, p_node_b, remained_ls,
			a_waiting_ls, b_waiting_ls, a_new_order, order_a, b_new_order, order_b);
		auto&& p_find_res = cache::Global_Cache<W>::p_cont_cache->find(key);
		if (p_find_res != cache::Global_Cache<W>::p_cont_cache->end()) {
			res = p_find_res->second;
			res.weight = res.weight * weight;
			return res;
		}
		else {
			double scale = 1.;

			auto&& remained_ls_pd = cache::pair_cmd();
			for (const auto& cmd : remained_ls) {
				if (cmd.first < order_a && cmd.second < order_b) {
					scale *= data_shape_a[cmd.first];
				}
				else {
					remained_ls_pd.push_back(cmd);
				}
			}

			auto&& a_waiting_ls_pd = cache::pair_cmd();
			for (const auto& cmd : a_waiting_ls) {
				if (cmd.first >= order_a) {
					a_waiting_ls_pd.push_back(cmd);
				}
			}

			auto&& b_waiting_ls_pd = cache::pair_cmd();
			for (const auto& cmd : b_waiting_ls) {
				if (cmd.first >= order_b) {
					b_waiting_ls_pd.push_back(cmd);
				}
			}

			auto&& not_operated = true;
			
			// note that the situation of both a,b being the waited index will not happen, because the iteration strategy updates either A or B at one time.
			if (!a_waiting_ls_pd.empty()) {
				if (order_a == a_waiting_ls_pd[0].first) {
					// w_node_a.node is not null in this case
					auto&& index_val = a_waiting_ls_pd[0].second;
					// close the waiting index
					auto&& next_a_waiting_ls = removed(a_waiting_ls_pd, 0);
					auto&& succ = p_node_a->get_successors()[index_val];
					res = contract_iterate(succ.node, p_node_b, succ.weight,
						data_shape_a, data_shape_b, remained_ls_pd, next_a_waiting_ls, b_waiting_ls_pd,
						a_new_order, b_new_order, parallel_tensor);
					not_operated = false;
				}
			}

			// the flag of whether node b is in the waiting list
			bool b_waited = false;
			if (!b_waiting_ls_pd.empty() && not_operated) {
				if (order_b == b_waiting_ls_pd[0].first) {
					// w_node_b.node is not null in this case
					auto&& index_val = b_waiting_ls_pd[0].second;
					// close the waiting index
					auto&& next_b_waiting_ls = removed(b_waiting_ls_pd, 0);
					auto&& succ = p_node_b->get_successors()[index_val];
					res = contract_iterate(p_node_a, succ.node, succ.weight,
						data_shape_a, data_shape_b, remained_ls_pd, a_waiting_ls_pd, next_b_waiting_ls,
						a_new_order, b_new_order, parallel_tensor);
					not_operated = false;
					b_waited = true;
				}
			}

			if (!remained_ls_pd.empty() && not_operated) {
				std::vector<node::weightednode<W>> new_successors;


				if (order_a >= remained_ls_pd[0].first) {
					auto&& next_remained_ls = removed(remained_ls_pd, 0);

					// find the right insert place in b_waiting_ls
					int insert_pos = get_cmd_insert_pos(b_waiting_ls_pd, remained_ls_pd[0].second);
					auto&& next_b_waiting_ls = inserted(b_waiting_ls_pd, insert_pos, std::make_pair(
						remained_ls_pd[0].second, 0)
					);

					new_successors = std::vector<node::weightednode<W>>(data_shape_a[remained_ls_pd[0].first]);
					if (order_a == remained_ls_pd[0].first) {
						// w_node_a.node is not null in this case
						auto&& successors_a = p_node_a->get_successors();
						int index_val = 0;
						auto&& i_new = new_successors.begin();
						for (auto&& i_a = successors_a.begin(); i_a != successors_a.end();
							i_new++, i_a++, index_val++) {
							next_b_waiting_ls[insert_pos].second = index_val;
							*i_new = contract_iterate(i_a->node, p_node_b, i_a->weight, data_shape_a, data_shape_b,
								next_remained_ls, a_waiting_ls_pd, next_b_waiting_ls,
								a_new_order, b_new_order, parallel_tensor);
						}
					}
					else {
						// this node skipped the opening index in this case
						int index_val = 0;
						for (auto&& succ_new : new_successors) {
							next_b_waiting_ls[insert_pos].second = index_val;
							succ_new = contract_iterate(p_node_a, p_node_b, wcomplex(1.,0.),
								data_shape_a, data_shape_b, next_remained_ls, a_waiting_ls_pd, next_b_waiting_ls,
								a_new_order, b_new_order, parallel_tensor
							);
							index_val++;
						}
					}
					not_operated = false;
				}
				else if (!b_waited) {
					// find the least opening index for node b (cause the remained list is sorted according to indices of a)
					auto&& next_to_open = min_iv(remained_ls_pd, &cmd_second_smaller);

					if (order_b >= next_to_open.second.second) {
						auto&& next_remained_ls = removed(remained_ls_pd, next_to_open.first);

						// find the right insert place in a_waiting_ls
						int insert_pos = get_cmd_insert_pos(a_waiting_ls_pd, next_to_open.second.first);
						auto&& next_a_waiting_ls = inserted(a_waiting_ls_pd, insert_pos, std::make_pair(
							next_to_open.second.first, 0)
						);

						new_successors = std::vector<node::weightednode<W>>(data_shape_b[next_to_open.second.second]);
						if (order_b == next_to_open.second.second) {
							// w_node_b.node is not null in this case
							auto&& successors_b = p_node_b->get_successors();
							int index_val = 0;
							auto&& i_new = new_successors.begin();
							for (auto&& i_b = successors_b.begin(); i_b != successors_b.end();
								i_new++, i_b++, index_val++) {
								next_a_waiting_ls[insert_pos].second = index_val;
								*i_new = contract_iterate(p_node_a, i_b->node, i_b->weight, data_shape_a, data_shape_b,
									next_remained_ls, next_a_waiting_ls, b_waiting_ls_pd, 
									a_new_order, b_new_order, parallel_tensor);
							}
						}
						else {
							// this node skipped the opening index in this case
							int index_val = 0;
							for (auto&& succ_new : new_successors) {
								next_a_waiting_ls[insert_pos].second = index_val;
								succ_new = contract_iterate(
									p_node_a, p_node_b, wcomplex(1.,0.),
									data_shape_a, data_shape_b, next_remained_ls, next_a_waiting_ls, b_waiting_ls_pd,
									a_new_order, b_new_order, parallel_tensor
								);
								index_val++;
							}
						}
						not_operated = false;
					}
				}

				// check whether sum up is needed
				if (!not_operated) {
					res = new_successors[0];
					for (auto&& i = new_successors.begin() + 1; i != new_successors.end(); i++) {
						res = sum(res, *i);
					}
				}
			}

			if (not_operated) {
				// in this case, no operation is performed, and nodes are weaved according to the rearrangement order.
				std::vector<node::weightednode<W>> new_successors;
				// arrange the node at this level				

				bool choice_A;
				if (p_node_a == nullptr) {
					choice_A = false;
				}
				else if (p_node_b == nullptr) {
					choice_A = true;
				}
				else {
					choice_A = a_new_order[order_a] < b_new_order[order_b];
				}

				if (choice_A) {
					// w_node_a.node will not be null
					auto&& successors_a = p_node_a->get_successors();
					new_successors = std::vector<node::weightednode<W>>(data_shape_a[order_a]);
					auto&& i_new = new_successors.begin();
					for (auto&& i_a = successors_a.begin(); i_a != successors_a.end(); i_new++, i_a++) {

						*i_new = contract_iterate(i_a->node, p_node_b, i_a->weight, data_shape_a, data_shape_b,
							remained_ls_pd, a_waiting_ls_pd, b_waiting_ls_pd,
							a_new_order, b_new_order, parallel_tensor);
					}
					auto&& temp_node = node::Node<W>(a_new_order[order_a], std::move(new_successors));
					res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
				}
				else {
					// w_node_b.node will not be null
					auto&& successors_b = p_node_b->get_successors();
					new_successors = std::vector<node::weightednode<W>>(data_shape_b[order_b]);
					auto&& i_new = new_successors.begin();
					for (auto&& i_b = successors_b.begin(); i_b != successors_b.end(); i_new++, i_b++) {
						*i_new = contract_iterate(p_node_a, i_b->node, i_b->weight, data_shape_a, data_shape_b,
							remained_ls_pd, a_waiting_ls_pd, b_waiting_ls_pd,
							a_new_order, b_new_order, parallel_tensor);
					}
					auto&& temp_node = node::Node<W>(b_new_order[order_b], std::move(new_successors));
					res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
				}

			}

			// add to the cache
			res.weight *= scale;
			cache::Global_Cache<W>::p_cont_cache->insert(std::make_pair(std::move(key), res));
			res.weight = res.weight * weight;
			return res;
		}
	}

	/// <summary>
	/// contract the designated indices, on w_node_a and w_node_b
	/// </summary>
	/// <param name="w_node_a"></param>
	/// <param name="w_node_b"></param>
	/// <param name="data_shape_a">given in the inner order</param>
	/// <param name="data_shape_b">given in the inner order</param>
	/// <param name="cont_indices">given in the inner order. list of (first, second). first: indices of a, second: indices of b </param>
	/// <param name="rearrangement">given in the inner order</param>
	/// <param name="parallel_tensor"></param>
	/// <returns></returns>
	static node::weightednode<W> contract(
		const node::weightednode<W>& w_node_a, const node::weightednode<W>& w_node_b,
		const std::vector<int64_t>& data_shape_a, const std::vector<int64_t>& data_shape_b,
		const cache::pair_cmd& cont_indices, 
		const std::vector<int64_t>& a_new_order,
		const std::vector<int64_t>& b_new_order,
		const std::vector<bool>& rearrangement, bool parallel_tensor) {

		// sort the remained_ls by first element, to keep the key unique
		cache::pair_cmd sorted_remained_ls(cont_indices);

		std::sort(sorted_remained_ls.begin(), sorted_remained_ls.end(),
			[](const std::pair<int, int>& a, const std::pair<int, int>& b) {
				return (a.first < b.first);
			});


		return contract_iterate(w_node_a.node, w_node_b.node, w_node_a.weight * w_node_b.weight,
			data_shape_a, data_shape_b, sorted_remained_ls, cache::pair_cmd(), cache::pair_cmd(),
			a_new_order, b_new_order, parallel_tensor);
	}
};