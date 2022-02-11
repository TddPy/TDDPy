#pragma once
#include "stdafx.h"
#include "Node.h"


namespace dict {

	//use id as the key
	typedef boost::unordered_map<int, CUDAcpl::Tensor> CUDAcpl_table;
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
}