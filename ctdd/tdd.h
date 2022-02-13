#pragma once
#include "stdafx.h"
#include "weighted_node.h"
#include "Node.h"

namespace tdd {

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

		/// <summary>
		/// Return the total shape and index_order after the reduction of specified indices.
		/// Note : Indices are counted in data indices only.
		/// </summary>
		/// <param name="length"></param>
		/// <param name="p_inner_i">corresponds to inner data indices, not the indices of tensor it represents.</param>
		/// <returns>[ownership transfer] a pointer to total shape</returns>
		std::pair<int64_t*, int*> index_reduced_info(int length, int* p_inner_i) const;

	public:


		/// <summary>
		/// print the informaton of this tdd. DEBUG usage.
		/// </summary>
		void __print() const;

		/// <summary>
		/// Note: All pointer ownership transfered.
		/// </summary>
		/// <param name="w_node"></param>
		/// <param name="dim_parallel"></param>
		/// <param name="dim_data"></param>
		/// <param name="p_index_order"></param>
		/// <param name="p_parallel_shape"></param>
		/// <param name="p_data_shape"></param>
		TDD(wnode::weightednode w_node, int dim_parallel, int dim_data, int* p_index_order, 
			int64_t* p_parallel_shape);

		TDD(const TDD& other);

		~TDD();

		// get the coresponding weighted node
		wnode::weightednode wnode() const;

		// get the total number of data indices
		int dim_data() const;
		
		// get the total number of parallel indices
		int dim_parallel() const;
		
		// get the shape of each data index
		const int64_t* data_shape() const;

		// get the shape of each parallel index
		const int64_t* parallel_shape() const;

		// get the data index order
		const int* index_order() const;

		/// <summary>
		/// Return the "inversion" of index order of this tdd.
		/// (it can be understand as the inverse in the permutation group.)
		/// </summary>
		/// <returns>[onwership transfer]</returns>
		int* inversed_order() const;


		/// <summary>
		/// return the corresponding inner data shape due to the index order
		/// </summary>
		/// <returns></returns>
		int64_t* inner_data_shape() const;


		// get the size of this tdd (defined as the number of different nodes)
		int get_size() const;

		TDD& operator = (const TDD& other);

		TDD clone() const;

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
		/// Sum up tdd a and b, and return the reduced result.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		static TDD sum(const TDD& a, const TDD& b);

		/// <summary>
		/// Contract the tdd according to the specified data_indices. Return the reduced result.
		/// data_indices should be counted in the data indices only.
		/// e.g. ([a, b, c], [d, e, f]) means contracting indices a - d, b - e, c - f(of course two lists should be in the same size)
		/// </summary>
		/// <param name="num_pair">number of pairs of tracing indices</param>
		/// <param name="p_i1">p_i1[i] less than p_i2[i] should hold for every i</param>
		/// <param name="p_i2">p_i1[i] less than p_i2[i] should hold for every i</param>
		/// <param name="p_sum_cache"></param>
		/// <returns></returns>
		TDD contract(int num_pair, const int* p_i1, const int* p_i2, dict::sum_table* p_sum_cache = nullptr);

		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="num_pair">number of pairs of contracting indices</param>
		/// <param name="p_ia">(of length num_pair)</param>
		/// <param name="p_ib">(of length num_pair)</param>
		/// <param name="parallel_tensor">Whether to tensor on the parallel indices.</param>
		/// <returns></returns>
		static TDD tensordot(const TDD& a, const TDD& b, int num_pair, const int* p_ia, const int* p_ib, bool parallel_tensor = false);

		
		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="num_indices">contract the last num_indices indices of a and first of b</param>
		/// <param name="parallel_tensor"></param>
		/// <returns></returns>
		static TDD tensordot(const TDD& a, const TDD& b, int num_indices, bool parallel_tensor = false);


		/// <summary>
		/// Transform this tensor to a CUDA complex and return.
		/// </summary>
		/// <returns></returns>
		CUDAcpl::Tensor CUDAcpl() const;
	};
}