#pragma once
#include "node.hpp"

namespace mng {
	inline void cache_clear_check();
	extern std::atomic<std::chrono::duration<double>> garbage_check_period;
}

namespace wnode {

	inline int get_cmd_insert_pos(const cache::pair_cmd& cmd_ls, int order) noexcept {
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

	template <class W>
	W get_normalizer(const node::succ_ls<W>& successors) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			int i_max = 0;
			double norm_max = norm(successors[0].weight);
			for (int i = 1; i < successors.size(); i++) {
				double temp_norm = norm(successors[i].weight);
				// alter the maximum according to EPS, to avoid arbitrary normalization
				if (temp_norm - norm_max > weight::EPS * norm_max) {
					i_max = i;
					norm_max = temp_norm;
				}
			}
			return successors[i_max].weight;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			auto&& sizes = successors[0].weight.sizes();
			auto norm_max = CUDAcpl::norm(successors[0].weight);

			auto&& normalizer = successors[0].weight.clone();

			for (int i = 1; i < successors.size(); i++) {
				auto temp_norm = CUDAcpl::norm(successors[i].weight);
				auto alter_matrix = temp_norm - norm_max > weight::EPS * norm_max;
				norm_max = torch::where(alter_matrix, temp_norm, norm_max);

				auto all_alter_matrix = alter_matrix.unsqueeze(alter_matrix.dim());
				all_alter_matrix = all_alter_matrix.expand_as(normalizer);
				normalizer = torch::where(all_alter_matrix, successors[i].weight, normalizer);
			}

			return normalizer;
		}
	}



	/// <summary>
	/// Get the new order, when reduced indices are removed, and the left indices in order.
	/// </summary>
	/// <param name="length"></param>
	/// <param name="reduced_indices">must be in order</param>
	/// <returns></returns>
	inline std::vector<int64_t> get_new_order(int length, const std::vector<int64_t>& reduced_indices) noexcept {
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

	template <class W>
	inline bool is_equal(const node::weightednode<W>& a, const node::weightednode<W>& b) noexcept {
		return (a.get_node() == b.get_node() && weight::is_equal(a.weight, b.weight));
	}

	/// <summary>
/// Conduct the normalization of this wnode.
/// This method only normalize the given wnode, and assumes the wnodes under it are already normalized.
/// (not terminal node)
/// </summary>
/// <returns>Return the normalized node and normalization coefficients as a wnode.</returns>
	template <class W>
	node::weightednode<W> normalize(const W& wei, int order, std::vector<node::weightednode<W>>&& successors) {

		// subnode equality check
		bool all_equal = true;
		for (auto p_succ = successors.begin() + 1; p_succ != successors.end(); p_succ++) {
			if (!is_equal(successors[0], *p_succ)) {
				all_equal = false;
				break;
			}
		}
		if (all_equal) {
			return node::weightednode<W>(
				weight::mul(wei, successors[0].weight),
				successors[0].get_node()
				);
		}

		// start to normalize the weights
		W weig_max = get_normalizer(successors);
		if (weight::is_exact_zero(weig_max)) {
			// in this case, all nodes in succesors should be the terminal node
			return node::weightednode<W>{ std::move(weig_max), nullptr };
		}

		W reciprocal = weight::reciprocal_without_zero(weig_max);

		for (auto&& succ : successors) {
			succ.weight = weight::mul(succ.weight, reciprocal);
			// check whether the successor weight is zero, and redirect to terminal node if so
			if (weight::is_zero(succ.weight)) {
				succ.weight = weight::zeros_like(wei);
				succ.set_node(nullptr);
			}
		}

		return node::weightednode<W>::get_wnode(weight::mul(weig_max, wei), order, std::move(successors));
	}


	/// <summary>
	/// To create the weighted node iteratively according to the instructions.
	/// </summary>
	/// <returns></returns>
	template <class W>
	node::weightednode<W> as_tensor_iterate(const CUDAcpl::Tensor& t,
		const std::vector<int64_t>& para_shape,
		const std::vector<int64_t>& data_shape,
		const std::vector<int64_t>& storage_order, int depth) {

		node::weightednode<W> res;
		// checks whether the tensor is reduced to the [[...[val]...]] form
		auto&& dim_data = data_shape.size() - 1;
		if (depth == dim_data) {
			weight::as_weight(t, res.weight, para_shape);
			res.set_node(nullptr);
			return res;
		}


		int split_pos = storage_order[depth];
		int axe_pos = t.dim() - dim_data + split_pos - 1;

		//note: torch::chunk does not work here

		auto new_successors = std::vector<node::weightednode<W>>(data_shape[split_pos]);
		for (int i = 0; i < data_shape[split_pos]; i++) {
			new_successors[i] = as_tensor_iterate<W>(
				// -1 is because the extra inner dim for real and imag
				t.select(axe_pos, i).unsqueeze(axe_pos),
				para_shape, data_shape, storage_order, depth + 1);
		}
		// normalize this depth
		res = normalize<W>(weight::ones<W>(para_shape), depth, std::move(new_successors));
		return res;
	}


	/// <summary>
	/// 
	/// </summary>
	/// <param name="w_node">Note that w_node.node should not be nullptr</param>
	/// <returns>tensor of dim (dim_data - node.order + 1)</returns>
	template <class W>
	CUDAcpl::Tensor to_CUDAcpl_iterate(const node::weightednode<W>& w_node,
		const std::vector<int64_t>& para_shape,
		const std::vector<int64_t>& data_shape) {
		// w_node.node is guaranteed not to be null

		auto&& current_order = w_node.get_node()->get_order();

		auto&& par_tensor = std::vector<CUDAcpl::Tensor>(w_node.get_node()->get_range());
		auto&& successors = w_node.get_node()->get_successors();

		auto dim_para = para_shape.size();
		auto&& dim_data = data_shape.size() - 1;


		CUDAcpl::Tensor temp_tensor;
		CUDAcpl::Tensor uniform_tensor;
		int next_order = 0;
		auto&& i_par = par_tensor.begin();
		for (auto&& i = successors.cbegin(); i != successors.cend(); i++, i_par++) {
			// detect terminal nodes, or iterate on the next node
			if (i->get_node() == nullptr) {
				temp_tensor = weight::from_weight(i->weight);
				next_order = dim_data;
			}
			else {
				cache::CUDAcpl_table_key<W> key;
				bool do_cache = i->is_multi_ref();
				if (do_cache) {
					// first look up in the dictionary
					key = cache::CUDAcpl_table_key<W>(i->get_node(), data_shape);
					//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
					cache::Global_Cache<W>::CUDAcpl_cache.first.lock_shared();
					auto&& p_find_res = cache::Global_Cache<W>::CUDAcpl_cache.second.find(key);
					if (p_find_res != cache::Global_Cache<W>::CUDAcpl_cache.second.end()) {
						uniform_tensor = p_find_res->second;
						cache::Global_Cache<W>::CUDAcpl_cache.first.unlock_shared();
						//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
					}
					else {
						cache::Global_Cache<W>::CUDAcpl_cache.first.unlock_shared();
						//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

						auto&& next_wnode = node::weightednode<W>(weight::ones<W>(para_shape), i->get_node());
						uniform_tensor = to_CUDAcpl_iterate(next_wnode, para_shape, data_shape);

						// add into the dictionary
						//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
						cache::Global_Cache<W>::CUDAcpl_cache.first.lock();
						cache::Global_Cache<W>::CUDAcpl_cache.second[key] = uniform_tensor;
						cache::Global_Cache<W>::CUDAcpl_cache.first.unlock();
						//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
					}
				}
				else {
					auto&& next_wnode = node::weightednode<W>(weight::ones<W>(para_shape), i->get_node());
					uniform_tensor = to_CUDAcpl_iterate(next_wnode, para_shape, data_shape);
				}
				next_order = i->get_node()->get_order();
				// multiply the dangling weight
				temp_tensor = weight::res_mul_weight(uniform_tensor, i->weight);
			}

			// broadcast according to the index distance
			if (next_order - current_order > 1) {
				// prepare the new data shape
				// The temp shape for adjustment.(shape of the successors of this node)
				auto&& p_temp_shape = std::vector<int64_t>(dim_para + dim_data - current_order);
				p_temp_shape[dim_para + dim_data - current_order - 1] = 2;
				for (int i = 0; i < dim_para; i++) {
					p_temp_shape[i] = para_shape[i];
				}
				for (int i = 0; i < next_order - current_order - 1; i++) {
					p_temp_shape[i + dim_para] = 1;
				}
				for (int i = dim_para; i < temp_tensor.dim() - 1; i++) {
					p_temp_shape[i + next_order - current_order - 1] = temp_tensor.size(i);
				}
				temp_tensor = temp_tensor.view(c10::IntArrayRef(p_temp_shape));
				for (int i = 0; i < next_order - current_order - 1; i++) {
					p_temp_shape[i + dim_para] = data_shape[i + current_order + 1];
				}
				temp_tensor = temp_tensor.expand(c10::IntArrayRef(p_temp_shape));
			}
			*i_par = std::move(temp_tensor);
		}
		auto&& res = torch::stack(par_tensor, dim_para);
		// multiply the dangling weight and return
		return weight::res_mul_weight(res, w_node.weight);
	}

	/// <summary>
	/// Get the CUDAcpl_Tensor determined from this node and the weights.
	///(use the trival index order)
	/// </summary>
	/// <param name="w_node"></param>
	/// <param name="inner_data_shape">data_shape(in the corresponding inner index order) is required, for the result should broadcast at reduced nodes of indices.
	/// Note that an *extra dimension* of 2 is needed at the end of p_inner_data_shape.</param>
	/// <returns></returns>
	template <class W>
	CUDAcpl::Tensor to_CUDAcpl(const node::weightednode<W>& w_node,
		const std::vector<int64_t>& para_shape,
		const std::vector<int64_t>& inner_data_shape) {
		int n_extra_one = 0;
		auto&& dim_data = inner_data_shape.size() - 1;
		CUDAcpl::Tensor res;
		if (w_node.get_node() == nullptr) {
			res = CUDAcpl::mul_element_wise(CUDAcpl::ones(para_shape), w_node.weight);
			n_extra_one = dim_data;
		}
		else {
			res = to_CUDAcpl_iterate(w_node, para_shape, inner_data_shape);
			n_extra_one = w_node.get_node()->get_order();
		}

		auto dim_para = para_shape.size();
		// this extra layer is for adding the reduced dimensions at the front
		// prepare the real data shape
		std::vector<int64_t> global_shape(para_shape);
		global_shape.insert(global_shape.end(), inner_data_shape.begin(), inner_data_shape.end());
		auto&& full_data_shape = std::vector<int64_t>(global_shape);
		for (int i = 0; i < n_extra_one; i++) {
			full_data_shape[dim_para + i] = 1;
		}
		res = res.view(c10::IntArrayRef(full_data_shape));
		res = res.expand(c10::IntArrayRef(global_shape));
		return res;
	}

	/// <summary>
	/// return the result of weightednode multiplied by the scalar (or tensor)
	/// </summary>
	/// <typeparam name="W1"></typeparam>
	/// <typeparam name="W2"></typeparam>
	/// <param name="w_node"></param>
	/// <param name="s"></param>
	/// <returns></returns>
	template <typename W1, typename W2>
	inline node::weightednode<weight::W_C<W1, W2>> operator *(const node::weightednode<W1>& w_node, const W2& s) {
		if constexpr (std::is_same_v<W1, wcomplex> && std::is_same_v<W2, CUDAcpl::Tensor>) {
			// not supported
		}
		else {
			auto new_weight = CUDAcpl::mul_element_wise(w_node.weight, s);

			// if the new weight is 0, then reduce all nodes.
			if (weight::is_exact_zero(new_weight)) {
				return node::weightednode<W1>(W1{ new_weight }, nullptr);
			}
			else {
				return node::weightednode<W1>(W1{ new_weight }, w_node.get_node());
			}
		}
	}

	template <typename W>
	node::weightednode<W> conj(const node::weightednode<W>& w_node) {
		node::Node<W>* new_node;

		if (w_node.get_node() == nullptr) {

			if constexpr (std::is_same_v<W, wcomplex>) {
				return node::weightednode<W>(std::conj(w_node.weight), nullptr);
			}
			else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
				return node::weightednode<W>(CUDAcpl::conj(w_node.weight), nullptr);
			}
		}
		else {
			auto&& successors = w_node.get_node()->get_successors();
			std::vector<node::weightednode<W>> new_successors(successors.size());
			for (int i = 0; i < successors.size(); i++) {
				new_successors[i] = wnode::conj(successors[i]);
			}

			if constexpr (std::is_same_v<W, wcomplex>) {
				return node::weightednode<W>::get_wnode(std::conj(w_node.weight), 
					w_node.get_node()->get_order(), std::move(new_successors));
			}
			else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
				return node::weightednode<W>::get_wnode(CUDAcpl::conj(w_node.weight),
					w_node.get_node()->get_order(), std::move(new_successors));
			}
		}

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
	template <class W>
	inline weight::sum_nweights<W> weights_normalize(const W& weight1, const W& weight2) {

		W renorm_coef;

		// first get renorm_coef
		if constexpr (std::is_same_v<W, wcomplex>) {
			auto norm1 = norm(weight1);
			auto norm2 = norm(weight2);
			renorm_coef = (norm2 - norm1 > weight::EPS * norm1) ? weight2 : weight1;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			auto norm2 = CUDAcpl::norm(weight2);
			auto norm1 = CUDAcpl::norm(weight1);
			auto chose_2 = norm2 - norm1 > weight::EPS * norm1;
			auto norm_max = torch::where(chose_2, norm2, norm1);

			auto all_chose_2 = chose_2.unsqueeze(chose_2.dim());
			all_chose_2 = all_chose_2.expand_as(weight1);

			renorm_coef = where(all_chose_2, weight2, weight1);
		}


		auto reciprocal = weight::reciprocal_without_zero(renorm_coef);

		auto nweight1 = weight::mul(weight1, reciprocal);
		auto nweight2 = weight::mul(weight2, reciprocal);

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
	template <class W>
	node::weightednode<W> sum_iterate(
		const node::weightednode<W>& w_node1,
		const node::weightednode<W>& w_node2, const W& renorm_coef, const std::vector<int64_t>& para_shape) {

		node::weightednode<W> res;
		if (w_node1.get_node() == nullptr && w_node2.get_node() == nullptr) {
			return node::weightednode<W>(weight::mul((w_node1.weight + w_node2.weight), renorm_coef), nullptr);
		}

		// produce the unique key and look up in the cache
		auto&& key = cache::sum_key<W>(
			w_node1.get_node(), w_node1.weight, w_node2.get_node(), w_node2.weight);

		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		cache::Global_Cache<W>::sum_cache.first.lock_shared();
		auto&& p_find_res = cache::Global_Cache<W>::sum_cache.second.find(key);
		auto found_in_cache = (p_find_res != cache::Global_Cache<W>::sum_cache.second.end());

		if (found_in_cache) {
			res = p_find_res->second.weightednode();
			cache::Global_Cache<W>::sum_cache.first.unlock_shared();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			res.weight = weight::mul(res.weight, renorm_coef);
			return res;
		}
		else {
			cache::Global_Cache<W>::sum_cache.first.unlock_shared();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				///////////////////////////////////////////////////////////////////////
			// ensure w_node1.node to be the node of smaller order, through swaping
			///////////////////////////////////////////////////////////////////////
			const node::weightednode<W>* p_wnode_1, * p_wnode_2;
			if (w_node1.get_node() == nullptr) {
				p_wnode_1 = &w_node2;
				p_wnode_2 = &w_node1;
			}
			else if (w_node2.get_node() == nullptr) {
				p_wnode_1 = &w_node1;
				p_wnode_2 = &w_node2;
			}
			else if (w_node1.get_node()->get_order() > w_node2.get_node()->get_order()) {
				p_wnode_1 = &w_node2;
				p_wnode_2 = &w_node1;
			}
			else {
				p_wnode_1 = &w_node1;
				p_wnode_2 = &w_node2;
			}

			auto&& new_successors = std::vector<node::weightednode<W>>(p_wnode_1->get_node()->get_range());

			bool not_operated = true;
			if (p_wnode_2->get_node() != nullptr) {

				// if they are the same node, then we can only adjust the weight
				if (p_wnode_1->get_node() == p_wnode_2->get_node()) {
					res = node::weightednode<W>(p_wnode_1->weight + p_wnode_2->weight, p_wnode_1->get_node());
					not_operated = false;
				}
				else if (p_wnode_1->get_node()->get_order() == p_wnode_2->get_node()->get_order()) {
					// node1 and node2 are assumed to have the same shape
					auto&& successors_1 = p_wnode_1->get_node()->get_successors();
					auto&& successors_2 = p_wnode_2->get_node()->get_successors();
					auto&& i_1 = successors_1.begin();
					auto&& i_2 = successors_2.begin();
					auto&& i_new = new_successors.begin();
					for (; i_1 != successors_1.end(); i_1++, i_2++, i_new++) {
						// normalize as a whole
						auto&& renorm_res = weights_normalize(
							weight::mul(p_wnode_1->weight, i_1->weight),
							weight::mul(p_wnode_2->weight, i_2->weight)
						);
						auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->get_node());
						auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), i_2->get_node());
						*i_new = sum_iterate<W>(next_wnode1, next_wnode2, renorm_res.renorm_coef, para_shape);
					}
					res = normalize<W>(weight::ones<W>(para_shape), p_wnode_1->get_node()->get_order(), std::move(new_successors));
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

				auto&& successors = p_wnode_1->get_node()->get_successors();
				auto&& i_1 = successors.begin();
				auto&& i_new = new_successors.begin();
				for (; i_1 != successors.end(); i_1++, i_new++) {
					auto&& renorm_res = weights_normalize(
						weight::mul(p_wnode_1->weight, i_1->weight), p_wnode_2->weight);
					auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), i_1->get_node());
					auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), p_wnode_2->get_node());
					*i_new = sum_iterate<W>(next_wnode1, next_wnode2, renorm_res.renorm_coef, para_shape);
				}
				res = normalize<W>(weight::ones<W>(para_shape), p_wnode_1->get_node()->get_order(), std::move(new_successors));
			}


			// cache the result
			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			cache::Global_Cache<W>::sum_cache.first.lock();
			cache::Global_Cache<W>::sum_cache.second[key] = res;
			cache::Global_Cache<W>::sum_cache.first.unlock();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			// multiply the renorm_coef and return
			res.weight = weight::mul(res.weight, renorm_coef);
			return res;
		}
	}

	/// <summary>
	/// sum two weighted nodes.
	/// </summary>
	/// <param name="w_node1"></param>
	/// <param name="w_node2"></param>
	/// <returns></returns>
	template <class W>
	node::weightednode<W> sum(
		const node::weightednode<W>& w_node1,
		const node::weightednode<W>& w_node2, const std::vector<int64_t>& para_shape) {
		// normalize as a whole
		auto&& renorm_res = weights_normalize(w_node1.weight, w_node2.weight);
		auto&& next_wnode1 = node::weightednode<W>(std::move(renorm_res.nweight1), w_node1.get_node());
		auto&& next_wnode2 = node::weightednode<W>(std::move(renorm_res.nweight2), w_node2.get_node());
		return sum_iterate<W>(next_wnode1, next_wnode2, renorm_res.renorm_coef, para_shape);
	}

	/// <summary>
	///	Note that :
	///	1. For remained_ls, we require smaller indices to be in the first place,
	///	and the corresponding larger one in the second place.
	///	2. indices in waiting_ls must be sorted in the asscending order to keep the cache key unique.
	/// </summary>
	/// <returns></returns>
	template <class W>
	node::weightednode<W> trace_iterate(const node::weightednode<W>& w_node,
		const std::vector<int64_t>& para_shape,
		const std::vector<int64_t>& data_shape,
		const cache::pair_cmd& remained_ls, const cache::pair_cmd& waiting_ls, const std::vector<int64_t>& new_order) {

		if (w_node.get_node() == nullptr) {
			// close all the unprocessed indices
			double scale = 1.;
			for (const auto& cmd : remained_ls) {
				scale *= data_shape[cmd.first];
			}
			return node::weightednode<W>(w_node.weight * scale, nullptr);
		}

		node::weightednode<W> res;

		// first look up in the dictionary
		auto&& key = cache::trace_key<W>(w_node.get_node(), remained_ls, waiting_ls);

		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		cache::Global_Cache<W>::trace_cache.first.lock_shared();
		auto&& p_find_res = cache::Global_Cache<W>::trace_cache.second.find(key);
		if (p_find_res != cache::Global_Cache<W>::trace_cache.second.end()) {
			res = p_find_res->second.weightednode();
			cache::Global_Cache<W>::trace_cache.first.unlock_shared();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			res.weight = weight::mul(res.weight, w_node.weight);
			return res;
		}
		else {
			cache::Global_Cache<W>::trace_cache.first.unlock_shared();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			auto&& order = w_node.get_node()->get_order();

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

			auto&& successors = w_node.get_node()->get_successors();

			if (!waiting_ls_pd.empty()) {
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
					res = trace_iterate(successors[index_val], para_shape, data_shape, remained_ls_pd, next_waiting_ls, new_order);
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
							if (i->get_node() == nullptr) {
								i_new->set_node(nullptr);
								i_new->weight = i->weight;
							}
							else {
								// adjust the new index value
								next_waiting_ls[insert_pos].second = index_val;
								*i_new = trace_iterate(*i, para_shape, data_shape, next_remained_ls, next_waiting_ls, new_order);
							}
						}
					}
					else {
						// this node skipped the index next_i_to_open in this case
						int index_val = 0;
						for (auto&& succ_new : new_successors) {
							next_waiting_ls[insert_pos].second = index_val;
							succ_new = trace_iterate(node::weightednode<W>(weight::ones<W>(para_shape), w_node.get_node()),
								para_shape, data_shape, next_remained_ls, next_waiting_ls, new_order);
							index_val++;
						}
					}

					// however the subnode outcomes are calculated, sum them over.
					res = new_successors[0];
					for (auto&& i = new_successors.begin() + 1; i != new_successors.end(); i++) {
						res = sum<W>(res, *i, para_shape);
					}
					not_operated = false;
				}
			}

			if (not_operated) {
				// in this case, no operation can be performed on this node, so we move on the the following nodes.
				auto&& new_successors = std::vector<node::weightednode<W>>(w_node.get_node()->get_range());
				auto&& i_new = new_successors.begin();
				for (auto&& i = successors.begin(); i != successors.end(); i++, i_new++) {
					if (i->get_node() == nullptr) {
						i_new->set_node(nullptr);
						i_new->weight = i->weight;
					}
					else {
						*i_new = trace_iterate(*i, para_shape, data_shape, remained_ls_pd, waiting_ls_pd, new_order);
					}
				}
				res = normalize<W>(weight::ones<W>(para_shape), new_order[order], std::move(new_successors));
			}

			res.weight = res.weight * scale;

			// add to the cache
			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			cache::Global_Cache<W>::trace_cache.first.lock();
			cache::Global_Cache<W>::trace_cache.second[key] = res;
			cache::Global_Cache<W>::trace_cache.first.unlock();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			res.weight = weight::mul(res.weight, w_node.weight);
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
	template <class W>
	node::weightednode<W> trace(const node::weightednode<W>& w_node,
		const std::vector<int64_t>& para_shape,
		const std::vector<int64_t>& data_shape,
		const cache::pair_cmd& remained_ls, const std::vector<int64_t> reduced_indices) {

		// sort the remained_ls by first element, to keep the key unique
		cache::pair_cmd sorted_remained_ls(remained_ls);

		std::sort(sorted_remained_ls.begin(), sorted_remained_ls.end(),
			[](const std::pair<int, int>& a, const std::pair<int, int>& b) {
				return (a.first < b.first);
			});

		auto&& num_pair = remained_ls.size();

		////////////////////////////////
		// prepare the new index order
		auto&& new_order = get_new_order(data_shape.size() - 1, reduced_indices);
		////////////////////////////////

		auto&& res = trace_iterate(w_node, para_shape, data_shape, sorted_remained_ls, cache::pair_cmd(), new_order);

		return res;
	}


	///////////////////////////////////////////////////////////////////////////////
	// Iteration Parallelism for cont
	namespace iter_para {
		extern ThreadPool* p_thread_pool;

		template <typename W1, typename W2>
		struct branch_state {
			int thread_count;
			node::weightednode<weight::W_C<W1, W2>> w_node;

			branch_state() noexcept {
				thread_count = 0;
			}
		};

		template <typename W1, typename W2>
		struct iter_state {
			std::vector<branch_state<W1, W2>> state;
			std::shared_mutex m;

			void init(int n) noexcept {
				state.resize(n);
			}
		};


		template <typename W1, typename W2>
		using para_coordinator = boost::unordered_map<cache::cont_key<W1, W2>, iter_state<W1, W2>>;

		template <typename W1, typename W2>
		struct Para_Crd {
			static para_coordinator<W1, W2> record;
			static std::shared_mutex m;
		};

		/// <summary>
		/// The mark that the corresponding branch have been calculated already.
		/// </summary>
		constexpr int CONT_DONE = (std::numeric_limits<int>::max)();
	}
	///////////////////////////////////////////////////////////////////////////////



	class iter_cont {
	public:
		/// <summary>
		/// The sub-program to decide the iteration strategy for non-parallel and parallel methods, respectively
		/// </summary>
		/// <typeparam name="W1"></typeparam>
		/// <typeparam name="W2"></typeparam>
		/// <param name="func"></param>
		/// <param name="key"></param>
		/// <param name="range"></param>
		template <typename W1, typename W2, typename FUNC>
		inline static void func(std::vector<node::weightednode<weight::W_C<W1, W2>>>& new_successors,
			const cache::cont_key<W1, W2>& key, int index_range, FUNC const& func) {
			// exam the parallel coordinator record
			iter_para::iter_state<W1, W2>* p_iter_state;
			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			iter_para::Para_Crd<W1, W2>::m.lock();
			auto&& p_find_record = iter_para::Para_Crd<W1, W2>::record.find(key);
			if (p_find_record == iter_para::Para_Crd<W1, W2>::record.end()) {
				iter_para::Para_Crd<W1, W2>::record[key].init(index_range);
				p_find_record = iter_para::Para_Crd<W1, W2>::record.find(key);
			}
			p_iter_state = &(p_find_record->second);
			iter_para::Para_Crd<W1, W2>::m.unlock();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			bool all_gathered = false;
			while (!all_gathered) {
				//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
				p_iter_state->m.lock();
				auto next_iter_index = 0;
				auto thread_count_min = p_iter_state->state[0].thread_count;
				all_gathered = p_iter_state->state[0].thread_count == iter_para::CONT_DONE;
				for (int i = 1; i < index_range; i++) {
					if (p_iter_state->state[i].thread_count != iter_para::CONT_DONE) {
						all_gathered = false;
						if (p_iter_state->state[i].thread_count < thread_count_min) {
							thread_count_min = p_iter_state->state[i].thread_count;
							next_iter_index = i;
						}
					}
				}

				// get the result, or conduct the next iteration
				if (all_gathered) {
					for (int i = 0; i < index_range; i++) {
						new_successors[i] = p_iter_state->state[i].w_node;
					}
					p_iter_state->m.unlock();
					//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				}
				else {
					p_iter_state->state[next_iter_index].thread_count += 1;
					p_iter_state->m.unlock();
					//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
					auto res = func(next_iter_index);

					//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
					p_iter_state->m.lock();

					p_iter_state->state[next_iter_index].thread_count = iter_para::CONT_DONE;
					p_iter_state->state[next_iter_index].w_node = res;

					p_iter_state->m.unlock();
					//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				}
			}
		}
	};


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
	template <typename W1, typename W2>
	node::weightednode<weight::W_C<W1, W2>> contract_iterate(
		const node::Node<W1>* p_node_a, const node::Node<W2>* p_node_b,
		const weight::W_C<W1, W2>& weight,
		const std::vector<int64_t>& para_shape_a, const std::vector<int64_t>& para_shape_b,
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
			return node::weightednode<weight::W_C<W1, W2>>(weight * scale, nullptr);
		}

		node::weightednode<weight::W_C<W1, W2>> res;

		auto&& order_a = p_node_a == nullptr ? data_shape_a.size() - 1 : p_node_a->get_order();
		auto&& order_b = p_node_b == nullptr ? data_shape_b.size() - 1 : p_node_b->get_order();
		// first look up in the dictionary
		// note that the results stored in cache is calculated with the uniformed (1., w_node_a.node) and (1., w_node_b.node)
		auto&& key = cache::cont_key<W1, W2>(p_node_a, p_node_b, remained_ls,
			a_waiting_ls, b_waiting_ls, a_new_order, order_a, b_new_order, order_b, parallel_tensor);

		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		// lock for parallelism
		cache::Cont_Cache<W1, W2>::cont_cache.first.lock_shared();
		auto&& p_find_res = cache::Cont_Cache<W1, W2>::cont_cache.second.find(key);
		auto found_in_cache = (p_find_res != cache::Cont_Cache<W1, W2>::cont_cache.second.end());

		if (found_in_cache) {
			res = p_find_res->second.weightednode();
			cache::Cont_Cache<W1, W2>::cont_cache.first.unlock_shared();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			res.weight = weight::mul(res.weight, weight);
			return res;
		}
		else {
			cache::Cont_Cache<W1, W2>::cont_cache.first.unlock_shared();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//std::cout << remained_ls << " / " << a_waiting_ls << " / " << b_waiting_ls << std::endl;

			double scale = 1.;

			bool a_node_uncontracted = true, b_node_uncontracted = true;
			// find out the next contracting index for B at the same time
			int next_b_min_i = 0;
			auto&& remained_ls_pd = cache::pair_cmd();
			for (const auto& cmd : remained_ls) {
				if (cmd.first < order_a && cmd.second < order_b) {
					scale *= data_shape_a[cmd.first];
				}
				else {
					if (cmd.first == order_a) {
						a_node_uncontracted = false;
					}
					if (cmd.second == order_b) {
						b_node_uncontracted = false;
					}
					remained_ls_pd.push_back(cmd);
					// record min i of b
					if (cmd.second < remained_ls_pd[next_b_min_i].second) {
						next_b_min_i = remained_ls_pd.size() - 1;
					}
				}
			}

			auto&& a_waiting_ls_pd = cache::pair_cmd();
			for (const auto& cmd : a_waiting_ls) {
				if (cmd.first >= order_a) {
					a_waiting_ls_pd.push_back(cmd);
					if (cmd.first == order_a) {
						a_node_uncontracted = false;
					}
				}
			}

			auto&& b_waiting_ls_pd = cache::pair_cmd();
			for (const auto& cmd : b_waiting_ls) {
				if (cmd.first >= order_b) {
					b_waiting_ls_pd.push_back(cmd);
					if (cmd.first == order_b) {
						b_node_uncontracted = false;
					}
				}
			}

			// first try to close the waited indices
			// note that the situation of both a,b being the waited index will not happen, because the iteration strategy updates either A or B at one time.
			if (!a_waiting_ls_pd.empty()) {
				if (order_a == a_waiting_ls_pd[0].first) {
					// w_node_a.node is not null in this case
					auto&& index_val = a_waiting_ls_pd[0].second;
					// close the waiting index
					auto&& next_a_waiting_ls = removed(a_waiting_ls_pd, 0);
					auto&& succ = p_node_a->get_successors()[index_val];
					res = contract_iterate<W1, W2>(succ.get_node(), p_node_b,
						weight::weight_expanded_back<W1, W2>(succ.weight, para_shape_b, parallel_tensor),
						para_shape_a, para_shape_b,
						data_shape_a, data_shape_b, remained_ls_pd, next_a_waiting_ls, b_waiting_ls_pd,
						a_new_order, b_new_order, parallel_tensor);
					goto RETURN;
				}
			}

			if (!b_waiting_ls_pd.empty()) {
				if (order_b == b_waiting_ls_pd[0].first) {
					// w_node_b.node is not null in this case
					auto&& index_val = b_waiting_ls_pd[0].second;
					// close the waiting index
					auto&& next_b_waiting_ls = removed(b_waiting_ls_pd, 0);
					auto&& succ = p_node_b->get_successors()[index_val];
					res = contract_iterate<W1, W2>(p_node_a, succ.get_node(),
						weight::weight_expanded_front<W1, W2>(succ.weight, para_shape_a, parallel_tensor),
						para_shape_a, para_shape_b,
						data_shape_a, data_shape_b, remained_ls_pd, a_waiting_ls_pd, next_b_waiting_ls,
						a_new_order, b_new_order, parallel_tensor);
					goto RETURN;
				}
			}


			// then try to weave the nodes of uncontracted indices
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
				if (a_node_uncontracted) {
					std::vector<node::weightednode<weight::W_C<W1, W2>>> new_successors;
					// w_node_a.node will not be null
					auto&& successors_a = p_node_a->get_successors();
					new_successors = std::vector<node::weightednode<weight::W_C<W1, W2>>>
						(data_shape_a[order_a]);

					iter_cont::func(new_successors, key, data_shape_a[order_a],
						[&](int i) {
							return contract_iterate<W1, W2>(successors_a[i].get_node(), p_node_b,
								weight::weight_expanded_back<W1, W2>(successors_a[i].weight, para_shape_b, parallel_tensor),
								para_shape_a, para_shape_b, data_shape_a, data_shape_b,
								remained_ls_pd, a_waiting_ls_pd, b_waiting_ls_pd,
								a_new_order, b_new_order, parallel_tensor);
						}
					);

					res = normalize<weight::W_C<W1, W2>>(
						weight::ones_like(weight),
						a_new_order[order_a],
						std::move(new_successors)
						);
					goto RETURN;
				}
			}
			else {
				if (b_node_uncontracted) {
					std::vector<node::weightednode<weight::W_C<W1, W2>>> new_successors;
					// w_node_b.node will not be null
					auto&& successors_b = p_node_b->get_successors();
					new_successors = std::vector<node::weightednode<weight::W_C<W1, W2>>>
						(data_shape_b[order_b]);

					iter_cont::func(new_successors, key, data_shape_b[order_b],
						[&](int i) {
							return contract_iterate<W1, W2>(p_node_a, successors_b[i].get_node(),
								weight::weight_expanded_front<W1, W2>(successors_b[i].weight, para_shape_a, parallel_tensor),
								para_shape_a, para_shape_b, data_shape_a, data_shape_b,
								remained_ls_pd, a_waiting_ls_pd, b_waiting_ls_pd,
								a_new_order, b_new_order, parallel_tensor);
						}
					);

					res = normalize<weight::W_C<W1, W2>>(
						weight::ones_like(weight),
						b_new_order[order_b],
						std::move(new_successors)
						);
					goto RETURN;
				}
			}



			// remained_ls not empty holds in this situation
			{
				std::vector<node::weightednode<weight::W_C<W1, W2>>> new_successors;

				if (order_a >= remained_ls_pd[0].first) {
					auto&& next_remained_ls = removed(remained_ls_pd, 0);

					// find the right insert place in b_waiting_ls
					int insert_pos = get_cmd_insert_pos(b_waiting_ls_pd, remained_ls_pd[0].second);
					auto&& next_b_waiting_ls = inserted(b_waiting_ls_pd, insert_pos, std::make_pair(
						remained_ls_pd[0].second, 0)
					);

					new_successors = std::vector<node::weightednode<weight::W_C<W1, W2>>>
						(data_shape_a[remained_ls_pd[0].first]);
					if (order_a == remained_ls_pd[0].first) {
						// w_node_a.node is not null in this case
						auto&& successors_a = p_node_a->get_successors();

						iter_cont::func(new_successors, key, data_shape_a[remained_ls_pd[0].first],
							[&](int i) {
								next_b_waiting_ls[insert_pos].second = i;
								return contract_iterate<W1, W2>(successors_a[i].get_node(), p_node_b,
									weight::weight_expanded_back<W1, W2>(successors_a[i].weight, para_shape_b, parallel_tensor),
									para_shape_a, para_shape_b, data_shape_a, data_shape_b,
									next_remained_ls, a_waiting_ls_pd, next_b_waiting_ls,
									a_new_order, b_new_order, parallel_tensor);
							}
						);
					}
					else {
						// this node skipped the opening index in this case
						iter_cont::func(new_successors, key, data_shape_a[remained_ls_pd[0].first],
							[&](int i) {
								next_b_waiting_ls[insert_pos].second = i;
								return contract_iterate<W1, W2>(p_node_a, p_node_b,
									weight::ones_like(weight), para_shape_a, para_shape_b,
									data_shape_a, data_shape_b, next_remained_ls, a_waiting_ls_pd, next_b_waiting_ls,
									a_new_order, b_new_order, parallel_tensor
									);
							}
						);
					}
				}
				else if (order_b >= remained_ls_pd[next_b_min_i].second) {
					auto&& next_remained_ls = removed(remained_ls_pd, next_b_min_i);

					// find the right insert place in a_waiting_ls
					int insert_pos = get_cmd_insert_pos(a_waiting_ls_pd, remained_ls_pd[next_b_min_i].first);
					auto&& next_a_waiting_ls = inserted(a_waiting_ls_pd, insert_pos, std::make_pair(
						remained_ls_pd[next_b_min_i].first, 0)
					);

					new_successors = std::vector<node::weightednode<weight::W_C<W1, W2>>>
						(data_shape_b[remained_ls_pd[next_b_min_i].second]);
					if (order_b == remained_ls_pd[next_b_min_i].second) {
						// w_node_b.node is not null in this case
						auto&& successors_b = p_node_b->get_successors();

						iter_cont::func(new_successors, key, data_shape_b[remained_ls_pd[next_b_min_i].second],
							[&](int i) {
								next_a_waiting_ls[insert_pos].second = i;
								return contract_iterate<W1, W2>(p_node_a, successors_b[i].get_node(),
									weight::weight_expanded_front<W1, W2>(successors_b[i].weight, para_shape_a, parallel_tensor),
									para_shape_a, para_shape_b, data_shape_a, data_shape_b,
									next_remained_ls, next_a_waiting_ls, b_waiting_ls_pd,
									a_new_order, b_new_order, parallel_tensor);
							}
						);
					}
					else {
						// this node skipped the opening index in this case
						iter_cont::func(new_successors, key, data_shape_b[remained_ls_pd[next_b_min_i].second],
							[&](int i) {
								next_a_waiting_ls[insert_pos].second = i;
								return contract_iterate<W1, W2>(
									p_node_a, p_node_b, weight::ones_like(weight), para_shape_a, para_shape_b,
									data_shape_a, data_shape_b, next_remained_ls, next_a_waiting_ls, b_waiting_ls_pd,
									a_new_order, b_new_order, parallel_tensor
									);
							}
						);
					}
				}

				// produce the shape for sum
				std::vector<int64_t> para_shape(para_shape_a);
				if (parallel_tensor) {
					para_shape.insert(para_shape.end(), para_shape_b.begin(), para_shape_b.end());
				}

				// sum up
				res = new_successors[0];
				for (auto&& i = new_successors.begin() + 1; i != new_successors.end(); i++) {
					res = sum<weight::W_C<W1, W2>>(res, *i, para_shape);
				}
			}


		RETURN:
			// add to the cache
			res.weight = res.weight * scale;

			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			cache::Cont_Cache<W1, W2>::cont_cache.first.lock();
			cache::Cont_Cache<W1, W2>::cont_cache.second[key] = res;
			cache::Cont_Cache<W1, W2>::cont_cache.first.unlock();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			res.weight = weight::mul(res.weight, weight);
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
	/// <param name="parallel_tensor"></param>
	/// <returns></returns>
	template <typename W1, typename W2>
	node::weightednode<weight::W_C<W1, W2>> contract(
		const node::weightednode<W1>& w_node_a, const node::weightednode<W2>& w_node_b,
		const std::vector<int64_t>& para_shape_a, const std::vector<int64_t>& para_shape_b,
		const std::vector<int64_t>& data_shape_a, const std::vector<int64_t>& data_shape_b,
		const cache::pair_cmd& cont_indices,
		const std::vector<int64_t>& a_new_order,
		const std::vector<int64_t>& b_new_order, bool parallel_tensor) {



		// sort the remained_ls by first element, to keep the key unique
		cache::pair_cmd sorted_remained_ls(cont_indices);

		std::sort(sorted_remained_ls.begin(), sorted_remained_ls.end(),
			[](const std::pair<int, int>& a, const std::pair<int, int>& b) {
				return (a.first < b.first);
			});


		std::vector<std::future<node::weightednode<weight::W_C<W1, W2>>>> results(iter_para::p_thread_pool->thread_num());

		auto prepared_weight = weight::prepare_weight(w_node_a.weight, w_node_b.weight, parallel_tensor);
		for (int i = 0; i < iter_para::p_thread_pool->thread_num(); i++) {
			results[i] = iter_para::p_thread_pool->enqueue(
				[&] {
					return contract_iterate<W1, W2>(w_node_a.get_node(), w_node_b.get_node(), prepared_weight,
						para_shape_a, para_shape_b,
						data_shape_a, data_shape_b, sorted_remained_ls, cache::pair_cmd(), cache::pair_cmd(),
						a_new_order, b_new_order, parallel_tensor);
				}
			);
		}

		while (results[0].wait_for(mng::garbage_check_period.load()) != std::future_status::ready) {
			mng::cache_clear_check();
		}

		auto&& res = results[0].get();

		for (int i = 1; i < results.size(); i++) {
			results[i].get();
		}

		iter_para::Para_Crd<W1, W2>::record.clear();

		return res;
	}
};