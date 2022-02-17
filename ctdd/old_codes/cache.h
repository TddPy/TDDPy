#pragma once
#include "stdafx.h"
#include "weights.h"

namespace node {
	template <class W>
	class Node;
}

namespace wnode {
	template <class W>
	struct weightednode;
}

namespace cache {

	// the type for unique table
	template <class W>
	struct unique_table_key {
		node_int order;
		node_int range;
		int* p_weights_real;
		int* p_weights_imag;
		//note that p_nodes will always be borrowed.
		const node::Node<W>** p_nodes;

		/// <summary>
		/// Construction of a unique_table key (complex version)
		/// </summary>
		/// <param name="_order"></param>
		/// <param name="_range"></param>
		/// <param name="_p_weights">[borrowed]</param>
		/// <param name="_p_nodes">[borrowed</param>
		unique_table_key(node_int _order, node_int _range,
			const wcomplex* _p_weights, const node::Node<wcomplex>** _p_nodes);
		/// <summary>
		/// COnstruction of a unique_table key (tensor version)
		/// </summary>
		/// <param name="_order"></param>
		/// <param name="_range"></param>
		/// <param name="_p_weights"></param>
		/// <param name="_p_nodes"></param>
		unique_table_key(node_int _order, node_int _range,
			const CUDAcpl::Tensor* _p_weights, const node::Node<CUDAcpl::Tensor>** _p_nodes);
		unique_table_key(const unique_table_key& other);
		unique_table_key& operator =(const unique_table_key& other);
		~unique_table_key();
	};

	bool operator == (const unique_table_key<wcomplex>& a, const unique_table_key<wcomplex>& b);
	bool operator == (const unique_table_key<CUDAcpl::Tensor>& a, const unique_table_key<CUDAcpl::Tensor>& b);

	std::size_t hash_value(const unique_table_key<wcomplex>& key_struct);
	std::size_t hash_value(const unique_table_key<CUDAcpl::Tensor>& key_struct);

	template <typename W>
	using unique_table = boost::unordered_map<unique_table_key<W>, const node::Node<W>*>;

	// the type for duplicate cache
	template <typename W>
	using duplicate_table = boost::unordered_map<int, const node::Node<W>*>;






	// the type for CUDAcpl cache
	template <class W>
	using CUDAcpl_table = boost::unordered_map<int, CUDAcpl::Tensor>;


	// the type for summation cache
	template <class  W>
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
	bool operator == (const sum_key<wcomplex>& a, const sum_key<wcomplex>& b);

	std::size_t hash_value(const sum_key<wcomplex>& key_struct);

	template <class W>
	using sum_table = boost::unordered_map<sum_key<W>, wnode::weightednode<W>>;


	// the type for contraction cache
	template <class W>
	struct cont_key {
		int id;
		int num_remained;
		int* p_r_i1;
		int* p_r_i2;
		int num_waiting;
		int* p_w_i;
		int* p_w_v;

		/// <summary>
		/// Note: all pointer ownership borrowed.
		/// </summary>
		/// <param name="_id"></param>
		/// <param name="_num_remained"></param>
		/// <param name="_p_r_i1"></param>
		/// <param name="_p_r_i2"></param>
		/// <param name="_num_waiting"></param>
		/// <param name="_p_w_i"></param>
		/// <param name="_p_w_v"></param>
		cont_key(int _id, int _num_remained, const int* _p_r_i1, const int* _p_r_i2,
			int _num_waiting, const int* _p_w_i, const int* _p_w_v);
		cont_key(const cont_key& other);
		cont_key<W>& operator =(const cont_key<W>& other);
		~cont_key();
	};

	template <class W>
	bool operator == (const cont_key<W>& a, const cont_key<W>& b);

	template <class W>
	std::size_t hash_value(const cont_key<W>& key_struct);

	template <class W>
	using cont_table = boost::unordered_map<cont_key<W>, wnode::weightednode<W>>;


	template <class W>
	struct Global_Cache {
		static duplicate_table<W> duplicate_cache;
		static duplicate_table<W> shift_cache;
		static CUDAcpl_table<W> CUDAcpl_cache;
		static sum_table<W> sum_cache;
		static cont_table<W> cont_cache;

		static void reset() {
			duplicate_cache = duplicate_table<W>();
			shift_cache = duplicate_table<W>();
			CUDAcpl_cache = CUDAcpl_table<W>();
			sum_cache = sum_table<W>();
			cont_cache = cont_table<W>();
		}
	};



	/*Implementation below*/


	template <class W>
	unique_table_key<W>::~unique_table_key() {
#ifdef DECONSTRUCTOR_DEBUG
		if (p_weights_real == nullptr || p_weights_imag == nullptr) {
			std::std::cout << "unique_table_key repeat deconstruction" << std::std::endl;
		}
		free(p_weights_real);
		p_weights_real = nullptr;
		free(p_weights_imag);
		p_weights_imag = nullptr;
#else
		free(p_weights_real);
		free(p_weights_imag);
#endif
	}


	template <class W>
	cache::cont_key<W>::cont_key(int _id, int _num_remained, const int* _p_r_i1, const int* _p_r_i2,
		int _num_waiting, const int* _p_w_i, const int* _p_w_v) {
		id = _id;
		num_remained = _num_remained;
		p_r_i1 = array_clone(_p_r_i1, num_remained);
		p_r_i2 = array_clone(_p_r_i2, num_remained);
		num_waiting = _num_waiting;
		p_w_i = array_clone(_p_w_i, num_waiting);
		p_w_v = array_clone(_p_w_v, num_waiting);
	}

	template <class W>
	cache::cont_key<W>::cont_key(const cont_key<W>& other) {
		id = other.id;
		num_remained = other.num_remained;
		p_r_i1 = array_clone(other.p_r_i1, num_remained);
		p_r_i2 = array_clone(other.p_r_i2, num_remained);
		num_waiting = other.num_waiting;
		p_w_i = array_clone(other.p_w_i, num_waiting);
		p_w_v = array_clone(other.p_w_v, num_waiting);
	}

	template <class W>
	cache::cont_key<W>& cache::cont_key<W>::operator =(const cont_key<W>& other) {
		id = other.id;
		if (num_remained != other.num_remained) {
			num_remained = other.num_remained;
			free(p_r_i1);
			free(p_r_i2);
			p_r_i1 = (int*)malloc(sizeof(int) * num_remained);
			p_r_i2 = (int*)malloc(sizeof(int) * num_remained);
		}
		if (num_waiting != other.num_waiting) {
			num_waiting = other.num_waiting;
			free(p_w_i);
			free(p_w_v);
			p_w_i = (int*)malloc(sizeof(int) * num_waiting);
			p_w_v = (int*)malloc(sizeof(int) * num_waiting);
		}
		for (int i = 0; i < num_remained; i++) {
			p_r_i1[i] = other.p_r_i1[i];
			p_r_i2[i] = other.p_r_i2[i];
		}
		for (int i = 0; i < num_waiting; i++) {
			p_w_i[i] = other.p_w_i[i];
			p_w_v[i] = other.p_w_v[i];
		}
		return *this;
	}

	template <class W>
	cache::cont_key<W>::~cont_key() {
#ifdef DECONSTRUCTOR_DEBUG
		if (p_r_i1 == nullptr || p_r_i2 == nullptr ||
			p_w_i == nullptr || p_w_v == nullptr) {
			std::std::cout << "cont_key repeat deconstruction" << std::std::endl;
		}
		free(p_r_i1);
		p_r_i1 = nullptr;
		free(p_r_i2);
		p_r_i2 = nullptr;
		free(p_w_i);
		p_w_i = nullptr;
		free(p_w_v);
		p_w_v = nullptr;
#else
		free(p_r_i1);
		free(p_r_i2);
		free(p_w_i);
		free(p_w_v);
#endif
	}

	template <class W>
	bool operator == (const cont_key<W>& a, const cont_key<W>& b) {
		if (a.id != b.id || a.num_remained != b.num_remained || a.num_waiting != b.num_waiting) {
			return false;
		}
		for (int i = 0; i < a.num_remained; i++) {
			if (a.p_r_i1[i] != b.p_r_i1[i] || a.p_r_i2[i] != b.p_r_i2[i]) {
				return false;
			}
		}
		for (int i = 0; i < a.num_waiting; i++) {
			if (a.p_w_i[i] != b.p_w_i[i] || a.p_w_v[i] != b.p_w_v[i]) {
				return false;
			}
		}
		return true;
	}

	template <class W>
	std::size_t hash_value(const cont_key<W>& key_struct) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key_struct.id);
		for (int i = 0; i < key_struct.num_remained; i++) {
			boost::hash_combine(seed, key_struct.p_r_i1[i]);
			boost::hash_combine(seed, key_struct.p_r_i2[i]);
		}
		for (int i = 0; i < key_struct.num_waiting; i++) {
			boost::hash_combine(seed, key_struct.p_w_i[i]);
			boost::hash_combine(seed, key_struct.p_w_v[i]);
		}
		return seed;
	}


}
