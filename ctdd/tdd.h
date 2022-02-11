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

		/* The total number of parallel indices in this TDD.
		* Note that the actual array is of dim_total+1 length, because there will be the extra inner dimension for CUDAcpl Wat the end.
		*/
		int m_dim_parallel;

		//The total number of data indices in this TDD.
		int m_dim_data;

		//The real index label of each inner data index.
		int* mp_index_order;

		//The shape of each parallel index.
		int64_t* mp_parallel_shape;

		//The shape of each data index.
		int64_t* mp_data_shape;


	private:

		/// <summary>
		/// get the corresponding shape for each inner index
		/// </summary>
		/// <param name="">[borrowed] Notice that the extra dimension of 2 at the end is needed.</param>
		void get_inner_data_shape(int64_t* p_storage) const;

	public:


		/// <summary>
		/// print the informaton of this tdd. DEBUG usage.
		/// </summary>
		void __print() const;

		~TDD();

		// get the coresponding weighted node
		wnode::weightednode wnode() const;

		// get the total number of data indices
		int dim_data() const;

		// get the shape of each data index)
		const int64_t* data_shape() const;

		// get the data index order
		const int* index_order() const;


		// get the size of this tdd (defined as the number of different nodes)
		int get_size() const;

		/// <summary>
		/// Construct a tdd tensor with the given direct representation.
		/// </summary>
		/// <param name="t">The given direct representation of a tensor.</param>
		/// <param name="dim_parallel">The number of parallel indices at front.</param>
		/// <param name="p_index_order">[borrowed] The index order used to stor this representation.
		/// Note that the item count + dim_parallel should be the dim of t strictly.
		/// If nullptr is put in, the trival order will be taken.</param>
		/// <returns>The tdd created.</returns>
		static TDD as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const int* p_index_order = nullptr);

		/// <summary>
		/// Directly return the cloned tdd.
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		static TDD as_tensor(const TDD& other);

		/// <summary>
		/// Return the direct product : a tensor b.The index order is the connection of that of a and b.
		///	
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="parallel_tensor">
		/// whether to tensor on the parallel indices. False : parallel index of aand b must be the same, and their shapes are :
		///	a: [(? ), (s_a), 2] tensor b : [(? ), (s_b), 2] ->[(? ), (s_a), (s_b), 2]
		/// True : tensor on the parallel indices too.Their shapes are :
		/// a: [(? a), (s_a), 2] tensor b : [(? b), (s_b), 2] ->[(? a), (? b), (s_a), (s_b), 2]
		/// </param>
		/// <returns></returns>
		static TDD direct_product(const TDD& a, const TDD& b, bool parallel_tensor = false);

		/// <summary>
		/// Transform this tensor to a CUDA complex and return.
		/// </summary>
		/// <returns></returns>
		CUDAcpl::Tensor CUDAcpl() const;
	};
}