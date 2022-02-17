#pragma once
#include "stdafx.h"
#include "weighted_node.h"

namespace tdd {

	//reset unique table and all cache
	void reset();

	/// <summary>
	/// TDD  functions as the compact representation of tensors,
	/// and can fit into tensor networks.
	/// </summary>
	template <class W>
	class TDD {
	private:
		//The inner data of this TDD.
		wnode::weightednode<W> m_wnode;

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
		TDD(wnode::weightednode<W> w_node, int dim_parallel, int dim_data, int* p_index_order,
			int64_t* p_parallel_shape);

		TDD(const TDD<W>& other);

		~TDD();

		// get the coresponding weighted node
		wnode::weightednode<W> wnode() const;

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

		TDD<W>& operator = (const TDD<W>& other);

		TDD<W> clone() const;

		/// <summary>
		/// Construct a tdd tensor with the given direct representation.
		/// </summary>
		/// <param name="t">The given direct representation of a tensor.</param>
		/// <param name="dim_parallel">The number of parallel indices at front.</param>
		/// <param name="p_index_order">[borrowed] The index order used to stor this representation.
		/// Note that the item count + dim_parallel should be the dim of t strictly.
		/// If nullptr is put in, the trival order will be taken.</param>
		/// <returns>The tdd created.</returns>
		static TDD<W> as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const int* p_index_order = nullptr);

		/// <summary>
		/// Directly return the cloned tdd.
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		static TDD<W> as_tensor(const TDD<W>& other);

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
		static TDD<W> direct_product(const TDD<W>& a, const TDD<W>& b, bool parallel_tensor = false);

		/// <summary>
		/// Sum up tdd a and b, and return the reduced result.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		static TDD<W> sum(const TDD<W>& a, const TDD<W>& b);

		/// <summary>
		/// Contract the tdd according to the specified data_indices. Return the reduced result.
		/// data_indices should be counted in the data indices only.
		/// e.g. ([a, b, c], [d, e, f]) means contracting indices a - d, b - e, c - f(of course two lists should be in the same size)
		/// </summary>
		/// <param name="num_pair">number of pairs of tracing indices</param>
		/// <param name="p_i1">p_i1[i] less than p_i2[i] should hold for every i</param>
		/// <param name="p_i2">p_i1[i] less than p_i2[i] should hold for every i</param>
		/// <returns></returns>
		TDD<W> contract(int num_pair, const int* p_i1, const int* p_i2);

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
		static TDD<W> tensordot(const TDD<W>& a, const TDD<W>& b, int num_pair, const int* p_ia, const int* p_ib, bool parallel_tensor = false);


		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="num_indices">contract the last num_indices indices of a and first of b</param>
		/// <param name="parallel_tensor"></param>
		/// <returns></returns>
		static TDD<W> tensordot(const TDD<W>& a, const TDD<W>& b, int num_indices, bool parallel_tensor = false);


		/// <summary>
		/// Transform this tensor to a CUDA complex and return.
		/// </summary>
		/// <returns></returns>
		CUDAcpl::Tensor CUDAcpl() const;
	};









	/*implementation below*/


	template <class W>
	void TDD<W>::get_inner_data_shape(int64_t* p_storage) const {
		for (int i = 0; i < m_dim_data; i++) {
			p_storage[i] = mp_data_shape[mp_index_order[i]];
		}
		p_storage[m_dim_data] = 2;
	}

	template <class W>
	std::pair<int64_t*, int*> TDD<W>::index_reduced_info(int length, int* p_inner_i) const {
		int* p_order = (int*)malloc(sizeof(int) * (m_dim_data - length));
		int64_t* p_shape = (int64_t*)malloc(sizeof(int64_t) * (m_dim_data + m_dim_parallel + 1 - length));
		p_shape[m_dim_data + m_dim_parallel - length] = 2;
		for (int i = 0; i < m_dim_parallel; i++) {
			p_shape[i] = mp_parallel_shape[i];
		}

		auto p_inner_shape = inner_data_shape();

		int new_count = 0;
		for (int i = 0; i < m_dim_data; i++) {
			// check whether this inner index is reduced
			bool reduced = false;
			for (int j = 0; j < length; j++) {
				if (i == p_inner_i[j]) {
					reduced = true;
					break;
				}
			}
			if (!reduced) {
				p_shape[new_count + m_dim_parallel] = p_inner_shape[i];
				p_order[new_count] = mp_index_order[i];
				new_count++;
			}
		}

		free(p_inner_shape);

		int* p_order_res = (int*)malloc(sizeof(int) * (m_dim_data - length));
		for (int i = 0; i < m_dim_data - length; i++) {
			p_order_res[i] = i;
		}
		std::sort(p_order_res, p_order_res + m_dim_data - length,
			[p_order](int a, int b) {
				return (p_order[a] < p_order[b]);
			}
		);
		int64_t* p_shape_res = array_clone(p_shape, m_dim_data + m_dim_parallel + 1 - length);
		//sort the data_shape
		for (int i = 0; i < m_dim_data - length; i++) {
			p_shape_res[m_dim_parallel + i] = p_shape[p_order_res[i]];
		}
		free(p_order);
		free(p_shape);
		return std::make_pair(p_shape_res, p_order_res);
	}

	template <class W>
	void TDD<W>::__print() const {
		std::cout << "weight: " << m_wnode.weight << std::endl;
		std::cout << "node: " << m_wnode.p_node << std::endl;
		std::cout << "dim parallel: " << m_dim_parallel << std::endl;
		std::cout << "parallel shape: (";
		for (int i = 0; i < m_dim_parallel; i++) {
			std::cout << mp_parallel_shape[i] << ", ";
		}
		std::cout << ")\n";
		std::cout << "dim data: " << m_dim_data << std::endl;
		std::cout << "data hape: (";
		for (int i = 0; i < m_dim_data; i++) {
			std::cout << mp_data_shape[i] << ", ";
		}
		std::cout << ")\n";
		std::cout << "index order: (";
		for (int i = 0; i < m_dim_data; i++) {
			std::cout << mp_index_order[i] << ", ";
		}
		std::cout << ")\n";
		std::cout << "size: " << get_size() << std::endl;
	}

	template <class W>
	TDD<W>::TDD(wnode::weightednode<W> w_node, int dim_parallel, int dim_data, int* p_index_order,
		int64_t* p_parallel_shape) {
		m_wnode = w_node;
		m_dim_parallel = dim_parallel;
		m_dim_data = dim_data;
		mp_index_order = p_index_order;
		mp_parallel_shape = p_parallel_shape;
		mp_data_shape = p_parallel_shape + m_dim_parallel;
	}

	template <class W>
	TDD<W>::TDD(const TDD<W>& other) {
		m_wnode = other.m_wnode;
		m_dim_parallel = other.m_dim_parallel;
		m_dim_data = other.m_dim_data;
		mp_index_order = array_clone(other.mp_index_order, m_dim_data);
		mp_parallel_shape = array_clone(other.mp_parallel_shape, m_dim_parallel + m_dim_data + 1);
		mp_data_shape = mp_parallel_shape + m_dim_parallel;
	}

	template <class W>
	TDD<W>::~TDD() {
		//mp_data_shape should not be freed because it points into mp_parallel_shape.
#ifdef DECONSTRUCTOR_DEBUG
		if (mp_index_order == nullptr || mp_parallel_shape == nullptr) {
			std::std::cout << "TDD repeat deconstruction" << std::std::endl;
		}
		free(mp_index_order);
		mp_index_order = nullptr;
		free(mp_parallel_shape);
		mp_parallel_shape = nullptr;
#else
		free(mp_index_order);
		free(mp_parallel_shape);
#endif
	}

	template <class W>
	wnode::weightednode<W> TDD<W>::wnode() const {
		return m_wnode;
	}

	template <class W>
	int TDD<W>::dim_data() const {
		return m_dim_data;
	}

	template <class W>
	int TDD<W>::dim_parallel() const {
		return m_dim_parallel;
	}

	template <class W>
	const int64_t* TDD<W>::data_shape() const {
		return mp_data_shape;
	}


	template <class W>
	const int64_t* TDD<W>::parallel_shape() const {
		return mp_parallel_shape;
	}

	template <class W>
	const int* TDD<W>::index_order() const {
		return mp_index_order;
	}

	template <class W>
	int* TDD<W>::inversed_order() const {
		int* p_res = (int*)malloc(sizeof(int) * m_dim_data);
		for (int i = 0; i < m_dim_data; i++) {
			p_res[mp_index_order[i]] = i;
		}
		return p_res;
	}

	template <class W>
	int64_t* TDD<W>::inner_data_shape() const {
		int64_t* p_res = (int64_t*)malloc(sizeof(int64_t) * m_dim_data);
		for (int i = 0; i < m_dim_data; i++) {
			p_res[i] = mp_data_shape[mp_index_order[i]];
		}
		return p_res;
	}


	template <class W>
	int TDD<W>::get_size() const {
		if (m_wnode.p_node == node::TERMINAL_NODE) {
			return 0;
		}
		return m_wnode.p_node->get_size();
	}

	template <class W>
	TDD<W>& TDD<W>::operator=(const TDD<W>& other) {
		if (mp_data_shape != nullptr) {
			free(mp_data_shape);
		}
		if (mp_parallel_shape != nullptr) {
			free(mp_parallel_shape);
		}
		if (mp_index_order != nullptr) {
			free(mp_index_order);
		}
		m_wnode = other.m_wnode;
		m_dim_data = other.m_dim_data;
		m_dim_parallel = other.m_dim_parallel;
		mp_data_shape = array_clone(other.mp_data_shape, m_dim_data);
		mp_index_order = array_clone(other.mp_index_order, m_dim_data);
		mp_parallel_shape = array_clone(other.mp_parallel_shape, m_dim_parallel);
		return *this;
	}

	template <class W>
	TDD<W> TDD<W>::clone() const {
		TDD* p_res = new TDD(*this);
		return *p_res;
	}

	template <class W>
	TDD<W> TDD<W>::as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const int* p_index_order) {
		auto dim_total = t.dim() - 1;
		auto dim_data = dim_total - dim_parallel;
		//use one int* to store parallel shape and data shape together.
		int64_t* p_parallel_shape = (int64_t*)malloc(sizeof(int64_t) * (dim_total + 1));
		//this is the extra inner dimension for CUDAcpl
		p_parallel_shape[dim_total] = 2;
		for (int i = 0; i < dim_total; i++) {
			p_parallel_shape[i] = t.size(i);
		}

		//prepare data_shape
		int64_t* p_data_shape = p_parallel_shape + dim_parallel;
		//prepare index_order
		int* p_index_order_pd = (int*)malloc(sizeof(int) * dim_data);
		if (p_index_order == nullptr) {
			for (int i = 0; i < dim_data; i++) {
				p_index_order_pd[i] = i;
			}
		}
		else {
			for (int i = 0; i < dim_data; i++) {
				p_index_order_pd[i] = p_index_order[i];
			}
		}

		wnode::weightednode<W> w_node = wnode::as_tensor_iterate<W>(t, dim_parallel, p_parallel_shape, dim_data, p_data_shape, p_index_order_pd, 0);

		//note the ownership transfer here
		TDD* p_res = new TDD(w_node, dim_parallel, dim_data, p_index_order_pd, p_parallel_shape);
		return *p_res;
	}

	template <class W>
	TDD<W> TDD<W>::as_tensor(const TDD<W>& other) {
		return other.clone();
	}

	template <class W>
	TDD<W> TDD<W>::direct_product(const TDD<W>& a, const TDD<W>& b, bool parallel_tensor) {
		if (parallel_tensor) {
			throw - 100;
		}
		else {
			// check the equality of parallel shapes
			//...
		}
		auto w_node = wnode::direct_product(a.m_wnode, a.m_dim_data, b.m_wnode, parallel_tensor);
		//It can be different for parallel situations
		auto dim_parallel = a.m_dim_parallel;
		auto dim_data = a.m_dim_data + b.m_dim_data;
		auto temp_index_order = (int*)malloc(sizeof(int) * (a.m_dim_data + b.m_dim_data));
		for (int i = 0; i < a.m_dim_data; i++) {
			temp_index_order[i] = a.mp_index_order[i];
		}
		for (int i = 0; i < b.m_dim_data; i++) {
			temp_index_order[i + a.m_dim_data] = b.mp_index_order[i] + a.m_dim_data;
		}
		auto p_parallel_shape = (int64_t*)malloc(sizeof(int64_t) * (dim_parallel + dim_data + 1));
		p_parallel_shape[dim_parallel + dim_data] = 2;
		for (int i = 0; i < dim_parallel; i++) {
			p_parallel_shape[i] = a.mp_parallel_shape[i];
		}
		for (int i = 0; i < a.m_dim_data; i++) {
			p_parallel_shape[i + dim_parallel] = a.mp_data_shape[i];
		}
		for (int i = 0; i < b.m_dim_data; i++) {
			p_parallel_shape[i + dim_parallel + a.m_dim_data] = b.mp_data_shape[i];
		}
		// +1 to put the extra dimension at the end.
		TDD* p_res = new TDD(w_node, dim_parallel, dim_data, temp_index_order, p_parallel_shape);
		return *p_res;
	}

	template <class W>
	TDD<W> TDD<W>::sum(const TDD<W>& a, const TDD<W>& b) {
		// check whether they are in the same shape
		//...
		auto res_wnode = wnode::sum(a.m_wnode, b.m_wnode, &cache::Global_Cache<W>::sum_cache);
		TDD&& res = a.clone();
		res.m_wnode = res_wnode;
		return res;
	}

	template <class W>
	TDD<W> TDD<W>::contract(int num_pair, const int* p_i1, const int* p_i2) {
		if (num_pair == 0) {
			return clone();
		}
		else {
			//transform to inner indices
			auto p_inversed_order = inversed_order();
			auto p_inner_data_shape = inner_data_shape();
			int* p_inner_i1 = (int*)malloc(sizeof(int) * num_pair);
			int* p_inner_i2 = (int*)malloc(sizeof(int) * num_pair);
			for (int i = 0; i < num_pair; i++) {
				p_inner_i1[i] = p_inversed_order[p_i1[i]];
				p_inner_i2[i] = p_inversed_order[p_i2[i]];
			}

			//note that inner_ls1[i] < inner_ls2[i] should hold for every i.
			auto&& res_wnode = wnode::contract(m_wnode, m_dim_data, p_inner_data_shape, num_pair, p_inner_i1, p_inner_i2);

			auto p_inner_i_reduced = array_concat(p_inner_i1, num_pair, p_inner_i2, num_pair);
			auto reduced_info = index_reduced_info(2 * num_pair, p_inner_i_reduced);
			free(p_inner_i_reduced);

			auto&& res_tdd = TDD(res_wnode, m_dim_parallel, m_dim_data - 2 * num_pair, reduced_info.second, reduced_info.first);

			free(p_inversed_order);
			free(p_inner_data_shape);
			free(p_inner_i1);
			free(p_inner_i2);
			return res_tdd;
		}
	}

	template <class W>
	TDD<W> TDD<W>::tensordot(const TDD<W>& a, const TDD<W>& b, int num_pair, const int* p_ia, const int* p_ib, bool parallel_tensor) {
		auto&& temp_tdd = direct_product(a, b, parallel_tensor);
		int* ptemp_ib = (int*)malloc(sizeof(int) * num_pair);
		for (int i = 0; i < num_pair; i++) {
			ptemp_ib[i] = p_ib[i] + a.m_dim_data;
		}
		auto&& res = temp_tdd.contract(num_pair, p_ia, ptemp_ib);
		free(ptemp_ib);
		return res;
	}

	template <class W>
	TDD<W> TDD<W>::tensordot(const TDD<W>& a, const TDD<W>& b, int num_indices, bool parallel_tensor) {
		int* p_ia = (int*)malloc(sizeof(int) * num_indices);
		int* p_ib = (int*)malloc(sizeof(int) * num_indices);
		for (int i = 0; i < num_indices; i++) {
			p_ia[i] = a.m_dim_data - num_indices + i;
			p_ib[i] = i;
		}
		TDD&& res = tensordot(a, b, num_indices, p_ia, p_ib, parallel_tensor);
		free(p_ia);
		free(p_ib);
		return res;
	}


	template <class W>
	CUDAcpl::Tensor TDD<W>::CUDAcpl() const {
		int64_t* p_inner_shape = (int64_t*)malloc(sizeof(int64_t) * (m_dim_data + 1));
		get_inner_data_shape(p_inner_shape);
		CUDAcpl::Tensor res = wnode::to_CUDAcpl(m_wnode, m_dim_data, p_inner_shape);
		free(p_inner_shape);
		return res;
	}

}