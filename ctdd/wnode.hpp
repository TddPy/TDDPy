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

		auto p_new_successors = std::vector<node::weightednode<W>>(data_shape[split_pos]);
		for (int i = 0; i < data_shape[split_pos]; i++) {
			p_new_successors[i] = as_tensor_iterate(split_tensor[i], parallel_shape, data_shape, index_order, depth + 1);
		}
		auto&& temp_node = node::Node<W>(depth, std::move(p_new_successors));
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
		auto&& p_new_successors = std::vector<node::weightednode<W>>(successors);
		for (auto i = p_new_successors.begin(); i != p_new_successors.end(); i++) {
			i->weight = i->weight / weig_max;
		}
		auto new_node = node::Node<W>::get_unique_node(w_node.node->get_order(), p_new_successors);
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
					cache::Global_Cache<W>::p_CUDAcpl_cache->insert(std::make_pair(key, uniform_tensor));
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
};