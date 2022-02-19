#pragma once
#include "node.hpp"


template <class W>
class wnode {
private:
	inline static bool cmd_smaller(const std::pair<int, int>& a, const std::pair<int, int>& b) {
		return a.first < b.first;
	}
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
	inline static weight::sum_nweights<W> sum_weights_normalize(const W& weight1, const W& weight2) {
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
			if (p_wnode_2->node != nullptr) {
				if (p_wnode_1->node->get_order() == p_wnode_2->node->get_order()) {
					// node1 and node2 are assumed to have the same shape
					auto&& successors_1 = p_wnode_1->node->get_successors();
					auto&& successors_2 = p_wnode_2->node->get_successors();
					auto&& i_1 = successors_1.begin();
					auto&& i_2 = successors_2.begin();
					auto&& i_new = new_successors.begin();
					for (; i_1 != successors_1.end(); i_1++, i_2++, i_new++) {
						// normalize as a whole
						auto&& renorm_res = sum_weights_normalize(p_wnode_1->weight * i_1->weight, p_wnode_2->weight * i_2->weight);
						auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->node);
						auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), i_2->node);
						*i_new = sum_iterate(next_wnode1, next_wnode2, renorm_res.renorm_coef);
					}
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
	static node::weightednode<W> sum(const node::weightednode<W>& w_node1, const node::weightednode<W>& w_node2) {
		// normalize as a whole
		auto&& renorm_res = sum_weights_normalize(w_node1.weight, w_node2.weight);
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
	static node::weightednode<W> contract_iterate(const node::weightednode<W>& w_node, const std::vector<int64_t>& data_shape,
		const cache::cont_cmd& remained_ls, const cache::cont_cmd& waiting_ls) {

		if (w_node.node == nullptr) {
			// close all the unprocessed indices
			double scale = 1.;
			for (const auto& cmd : waiting_ls) {
				scale *= data_shape[cmd.first];
			}
			return node::weightednode<W>(w_node.weight * scale, nullptr);
		}

		node::weightednode<W> res;

		// first look up in the dictionary
		auto&& key = cache::cont_key<W>(w_node.node->get_id(), remained_ls, waiting_ls);
		auto&& p_find_res = cache::Global_Cache<W>::p_cont_cache->find(key);
		if (p_find_res != cache::Global_Cache<W>::p_cont_cache->end()) {
			res = p_find_res->second;
			res.weight = res.weight * w_node.weight;
			return res;
		}
		else {
			auto&& order = w_node.node->get_order();

			// store the scaling number due to skipped remained indices
			double scale = 1.;

			// process the skipped remained indices (situations only first index skipped will be processed afterwards)
			auto&& remained_ls_pd = cache::cont_cmd();
			for (const auto& cmd : remained_ls) {
				if (cmd.second >= order) {
					remained_ls_pd.push_back(cmd);
				}
				else {
					scale *= data_shape[cmd.first];
				}
			}
			auto&& waiting_ls_pd = cache::cont_cmd();
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
				res = node::weightednode<W>(wcomplex(1., 0.) * scale, w_node.node);
				not_operated = false;
			}
			else if (!waiting_ls_pd.empty()) {
				/*
				*   waiting_ils is not empty in this case
				*	If multiple waiting indices have been skipped, we will resolve with iteration, one by one.
				*/

				// next_to_close: <pos in waiting_ls_pd, min in waiting_ls_pd>
				auto&& next_to_close = min_iv(waiting_ls_pd, &cmd_smaller);

				// note that next_i_to_close >= node.order is guaranteed here
				if (order == next_to_close.second.first) {
					auto&& index_val = waiting_ls_pd[next_to_close.first].second;
					// close the waiting index
					auto&& next_waiting_ls = removed<std::pair<int, int>>(waiting_ls_pd, next_to_close.first);
					res = contract_iterate(successors[index_val], data_shape, remained_ls_pd, next_waiting_ls);
					res.weight = res.weight * scale;
					not_operated = false;
				}
			}
			if (!remained_ls_pd.empty() && not_operated) {
				/*
				*	Check the remained indices to start tracing.
				*	If multiple (smaller ones of) remained indices have been skipped, we will resolve with iteration, one by one.
				*/
				auto&& next_to_open = min_iv(remained_ls_pd, &cmd_smaller);
				if (order >= next_to_open.second.first) {
					// open the index and finally sum up

					auto&& next_remained_ls = removed(remained_ls_pd, next_to_open.first);

					// find the right insert place in waiting_ls
					int insert_pos = 0;
					for (; insert_pos < waiting_ls_pd.size(); insert_pos++) {
						if (waiting_ls_pd[insert_pos].first < next_to_open.second.first) {
							continue;
						}
						else {
							break;
						}
					}

					// get the index range
					int range = data_shape[next_to_open.second.first];

					// produce the sorted new index lists
					auto&& next_waiting_ls = inserted(waiting_ls_pd, insert_pos, std::make_pair(
						remained_ls_pd[next_to_open.first].second, 0)
					);

					auto&& new_successors = std::vector<node::weightednode<W>>(range);
					if (order == next_to_open.second.first) {
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
								*i_new = contract_iterate(*i, data_shape, next_remained_ls, next_waiting_ls);
							}
						}
					}
					else {
						// this node skipped the index next_i_to_open in this case
						int index_val = 0;
						for (auto&& succ_new : new_successors) {
							next_waiting_ls[insert_pos].second = index_val;
							succ_new = contract_iterate(node::weightednode<W>(wcomplex(1., 0.), w_node.node),
								data_shape, next_remained_ls, next_waiting_ls);
							index_val++;
						}
					}

					// however the subnode outcomes are calculated, sum them over.
					res = new_successors[0];
					for (auto&& i = new_successors.begin() + 1; i != new_successors.end(); i++) {
						res = sum(res, *i);
					}
					res.weight = res.weight * scale;
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
						*i_new = contract_iterate(*i, data_shape, remained_ls_pd, waiting_ls_pd);
					}
				}
				auto&& temp_node = node::Node<W>(order, std::move(new_successors));
				res = normalize(node::weightednode<W>(wcomplex(1., 0.), &temp_node));
			}

			// add to the cache
			cache::Global_Cache<W>::p_cont_cache->insert(std::make_pair(std::move(key), res));
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
	static node::weightednode<W> contract(const node::weightednode<W>& w_node, const std::vector<int64_t>& data_shape,
		const cache::cont_cmd& remained_ls, const std::vector<int64_t> reduced_indices) {

		auto&& res = contract_iterate(w_node, data_shape, remained_ls, cache::cont_cmd());

		// shift the nodes at a time
		auto&& num_pair = remained_ls.size();

		////////////////////////////////
		// prepare the new index order
		auto&& new_order = std::vector<int64_t>(data_shape.size() - 1);
		for (int i = 0; i < reduced_indices[0]; i++) {
			new_order[i] = i;
		}
		for (int i = 0; i < num_pair * 2 - 1; i++) {
			for (int j = reduced_indices[i] + 1; j < reduced_indices[i + 1]; j++) {
				new_order[j] = j - i - 1;
			}
		}
		for (int j = reduced_indices[num_pair * 2 - 1] + 1; j < data_shape.size() - 1; j++) {
			new_order[j] = j - 2 * num_pair;
		}
		////////////////////////////////

		res.node = node::Node<W>::shift_multiple(res.node, new_order);
		return res;
	}

};