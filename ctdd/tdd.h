#pragma once
#include "stdafx.h"
#include "weighted_node.h"
#include "Node.h"

namespace tdd {
	/*
		Return the "inverse" of the given index order.
		(it can be understand as the inverse in the permutation group.)
	*/
	int* order_inverse(node_int index_count, const int* const index_order);

	/// <summary>
	/// TDD  functions as the compact representation of tensors,
	/// and can fit into tensor networks.
	/// </summary>
	class TDD {
	private:
		//The inner data of this TDD.
		wnode::weightednode m_wnode;

		//The total number of parallel indices in this TDD.
		int m_dim_parallel;

		//The total number of data indices in this TDD.
		int m_dim_data;

		//The real index label of each inner data index.
		int* mp_index_order;

		//The shape of each parallel index.
		int* mp_parallel_shape;

		//The shape of each data index.
		int* mp_data_shape;


	private:


	public:
		~TDD();

		// get the coresponding weighted node
		wnode::weightednode wnode() const;

		// get the total number of data indices
		int dim_data() const;

		// get the shape of each data index)
		const int* data_shape() const;

		// get the data index order
		const int* index_order() const;


		// get the size of this tdd (defined as the number of different nodes)
		int get_size() const;

		/// <summary>
		/// Construct a tdd tensor with the given direct representation.
		/// </summary>
		/// <param name="t">The given direct representation of a tensor.</param>
		/// <param name="dim_parallel">The number of parallel indices at front.</param>
		/// <param name="index_order">[borrowed] The index order used to stor this representation.
		/// Note that the item count + dim_parallel should be the dim of t strictly.
		/// If nullptr is put in, the trival order will be taken.</param>
		/// <returns>The tdd created.</returns>
		static TDD *as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const int* p_index_order = nullptr);

		/// <summary>
		/// Directly return the cloned tdd.
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		static TDD *as_tensor(const TDD& other);
	};
}