#pragma once
#include "stdafx.h"
#include "Node.h"

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
		int dim_parallel, const int* p_parallel_shape,
		int dim_data, const int* p_data_shape, const int* p_index_order, int depth);



	/// <summary>
	/// Conduct the normalization of this wnode.
	/// This method only normalize the given wnode, and assumes the wnodes under it are already normalized.
	/// </summary>
	/// <param name="w_node"></param>
	/// <returns>Return the normalized node and normalization coefficients as a wnode.</returns>
	weightednode normalize(weightednode w_node);
}