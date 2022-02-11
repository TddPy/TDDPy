#pragma once
#include "stdafx.h"
#include "Node.h"


namespace wnode {
	struct weightednode;
}

namespace dict {

	// the type for CUDAcpl cache
	typedef boost::unordered_map<int, CUDAcpl::Tensor> CUDAcpl_table;

	// the type for summation cache
	struct sum_key {
		int id_1;
		int nweight1_real;
		int nweight1_imag;
		int id_2;
		int nweight2_real;
		int nweight2_imag;

		/// <summary>
		/// Construct the key. Note that id_1 will be set as the smaller one.
		/// </summary>
		/// <param name="id_a"></param>
		/// <param name="weight_a"></param>
		/// <param name="id_b"></param>
		/// <param name="weight_b"></param>
		sum_key(int id_a, wcomplex weight_a, int id_b, wcomplex weight_b);
	};
	bool operator == (const sum_key& a, const sum_key& b);
	std::size_t hash_value(const sum_key& key_struct);

	typedef boost::unordered_map<sum_key, wnode::weightednode> sum_table;

}

namespace wnode {

	// The object of a weighted node.
	struct weightednode {
		wcomplex weight;
		const node::Node* p_node;
	};

	/// <summary>
	/// Check whether the two weighted nodes are equal.
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns></returns>
	inline bool is_equal(weightednode a, weightednode b);

	/// <summary>
	/// To create the weighted node iteratively according to the instructions.
	/// </summary>
	/// <param name="t"></param>
	/// <param name="dim_parallel"></param>
	/// <param name="p_parallel_shape"></param>
	/// <param name="dim_data"></param>
	/// <param name="p_data_shape"></param>
	/// <param name="p_index_order"></param>
	/// <param name="depth"></param>
	/// <returns></returns>
	weightednode as_tensor_iterate(const CUDAcpl::Tensor& t,
		int dim_parallel, const int64_t* p_parallel_shape,
		int dim_data, const int64_t* p_data_shape, const int* p_index_order, int depth);



	/// <summary>
	/// Conduct the normalization of this wnode.
	/// This method only normalize the given wnode, and assumes the wnodes under it are already normalized.
	/// </summary>
	/// <param name="w_node"></param>
	/// <returns>Return the normalized node and normalization coefficients as a wnode.</returns>
	weightednode normalize(weightednode w_node);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="w_node">Note that w_node.p_node should not be TERMINAL_NODE</param>
	/// <param name="dim_data"></param>
	/// <param name="p_data_shape"></param>
	/// <param name="tensor_cache">caches the corresponding tensor of this node (weights = 1)</param>
	/// <returns></returns>
	CUDAcpl::Tensor to_CUDAcpl_iterate(weightednode w_node, int dim_data, int64_t* p_data_shape, dict::CUDAcpl_table& tensor_cache);

	/// <summary>
	/// Get the CUDAcpl_Tensor determined from this node and the weights.
	///(use the trival index order)
	/// </summary>
	/// <param name="w_node"></param>
	/// <param name="dim_data"></param>
	/// <param name="p_inner_data_shape">[borrowed] data_shape(in the corresponding inner index order) is required, for the result should broadcast at reduced nodes of indices.
	/// Note that an *extra dimension* of 2 is needed at the end of p_inner_data_shape.</param>
	/// <returns></returns>
	CUDAcpl::Tensor to_CUDAcpl(weightednode w_node, int dim_data, int64_t* p_inner_data_shape);


	weightednode direct_product(weightednode a, int a_depth, weightednode b, bool parallel_tensor = false);

	struct sum_nweights {
		wcomplex nweight1;
		wcomplex nweight2;
		wcomplex renorm_coef;
	};
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
	sum_nweights sum_weights_normalize(wcomplex weight1, wcomplex weight2);

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
	weightednode sum_iterate(weightednode w_node1, weightednode w_node2, wcomplex renorm_coef, dict::sum_table sum_cache);

	/// <summary>
	/// sum two weighted nodes.
	/// </summary>
	/// <param name="w_node1"></param>
	/// <param name="w_node2"></param>
	/// <param name="p_sum_cache">[borrowed] used to cache the results. If nullptr, then a local cache will be used.</param>
	/// <returns></returns>
	weightednode sum(weightednode w_node1, weightednode w_node2, dict::sum_table* p_sum_cache = nullptr);
}