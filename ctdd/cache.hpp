#pragma once
#include "stdafx.h"
#include "weight.hpp"
#include "succ_ls.hpp"

namespace node {
	template <class W>
	class Node;
	template <class W>
	struct weightednode;
}


namespace cache {
	// the type for unique table
	template <class W>
	struct unique_table_key {
		int order;
		// real for <weight>, data for <tensor> (encode  in sequence)
		std::vector<weight::WCode> code1;
		// imag for <weight>, shape for <tensor>
		std::vector<weight::WCode> code2;
		std::vector<const node::Node<W>*> nodes;

		/// <summary>
		/// Construction of a unique_table key (complex version)
		/// </summary>
		/// <param name="_order"></param>
		/// <param name="_range"></param>
		/// <param name="_p_weights">[borrowed]</param>
		/// <param name="_p_nodes">[borrowed</param>
		unique_table_key(int _order, const node::succ_ls<W>& successors);

		unique_table_key(const unique_table_key& other) {
			order = other.order;
			code1 = other.code1;
			code2 = other.code2;
			nodes = other.nodes;
		}

		unique_table_key& operator =(unique_table_key&& other) {
			order = other.order;
			code1 = std::move(other.code1);
			code2 = std::move(other.code2);
			nodes = std::move(other.nodes);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const unique_table_key<W>& a, const unique_table_key<W>& b) {
		if (a.order != b.order) {
			return false;
		}
		// compare code2 first, because for <tensor> case it stores the shape information
		if (a.code2 != b.code2) {
			return false;
		}
		if (a.code1 != b.code1) {
			return false;
		}
		if (a.nodes != b.nodes) {
			return false;
		}
		return true;
	}

	template <class W>
	inline std::size_t hash_value(const unique_table_key<W>& key) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.order);
		for (const auto& code : key.code1) {
			boost::hash_combine(seed, code);
		}
		for (const auto& code : key.code2) {
			boost::hash_combine(seed, code);
		}
		for (const auto& code : key.nodes) {
			boost::hash_combine(seed, code);
		}
		return seed;
	}

	template <typename W>
	using unique_table = boost::unordered_map<unique_table_key<W>, const node::Node<W>*>;

	/* the type for duplicate cache
	*  first: id, second: order_shift
	*/
	template <typename W>
	using duplicate_table = boost::unordered_map<std::pair<int, int>, const node::Node<W>*>;


	template <class W>
	struct append_table_key {
		int id_a;
		int id_b;
		append_table_key(int _id_a, int _id_b) {
			id_a = _id_a;
			id_b = _id_b;
		}
	};


	template <class W>
	inline bool operator ==(const append_table_key<W>& a, const append_table_key<W>& b) {
		return a.id_a == b.id_a && a.id_b == b.id_b;
	}

	template <class W>
	inline std::size_t hash_value(const append_table_key<W>& key) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.id_a);
		boost::hash_combine(seed, key.id_b);
		return seed;
	}

	// the type for append cache
	template <typename W>
	using append_table = boost::unordered_map<append_table_key<W>, const node::Node<W>*>;




	template <class W>
	struct CUDAcpl_table_key {
		int id;
		std::vector<int64_t> inner_shape;

		/// <summary>
		/// Construction of a unique_table key (complex version)
		/// </summary>
		/// <param name="_order"></param>
		/// <param name="_range"></param>
		/// <param name="_p_weights">[borrowed]</param>
		/// <param name="_p_nodes">[borrowed</param>
		CUDAcpl_table_key(int _id, const std::vector<int64_t>& _inner_shape) {
			id = _id;
			inner_shape = _inner_shape;
		}

		CUDAcpl_table_key(const CUDAcpl_table_key& other) {
			id = other.id;
			inner_shape = other.inner_shape;
		}

		CUDAcpl_table_key& operator =(CUDAcpl_table_key&& other) {
			id = other.id;
			inner_shape = std::move(other.inner_shape);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const CUDAcpl_table_key<W>& a, const CUDAcpl_table_key<W>& b) {
		return a.id == b.id && a.inner_shape == b.inner_shape;
	}

	template <class W>
	inline std::size_t hash_value(const CUDAcpl_table_key<W>& key) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.id);
		for (const auto& code : key.inner_shape) {
			boost::hash_combine(seed, code);
		}
		return seed;
	}

	// the type for CUDAcpl cache
	template <class W>
	using CUDAcpl_table = boost::unordered_map<CUDAcpl_table_key<W>, CUDAcpl::Tensor>;


	/// <summary>
	/// the type for summation cache
	/// code1: real for wcomplex, data for tensor
	/// code2: imag for wcomplex, shape for tensor
	/// </summary>
	/// <typeparam name="W"></typeparam>
	template <class  W>
	struct sum_key {
		int id_1;
		std::vector<weight::WCode> nweight1_code1;
		std::vector<weight::WCode> nweight1_code2;
		int id_2;
		std::vector<weight::WCode> nweight2_code1;
		std::vector<weight::WCode> nweight2_code2;

		/// <summary>
		/// Construct the key. Note that id_1 will be set as the smaller one.
		/// </summary>
		/// <param name="id_a"></param>
		/// <param name="weight_a"></param>
		/// <param name="id_b"></param>
		/// <param name="weight_b"></param>
		sum_key(int id_a, const W& weight_a, int id_b, const W& weight_b);
		
		sum_key(const sum_key& other) {
			id_1 = other.id_1;
			nweight1_code1 = other.nweight1_code1;
			nweight1_code2 = other.nweight1_code2;
			id_2 = other.id_2;
			nweight2_code1 = other.nweight2_code1;
			nweight2_code2 = other.nweight2_code2;
		}

		sum_key& operator =(sum_key&& other) {
			id_1 = other.id_1;
			nweight1_code1 = std::move(other.nweight1_code1);
			nweight1_code2 = std::move(other.nweight1_code2);
			id_2 = other.id_2;
			nweight2_code1 = std::move(other.nweight2_code1);
			nweight2_code2 = std::move(other.nweight2_code2);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const sum_key<W>& a, const sum_key<W>& b) {
		return a.id_1 == b.id_1 && a.id_2 == b.id_2 &&
			a.nweight1_code1 == b.nweight1_code1 &&
			a.nweight1_code2 == b.nweight1_code2 &&
			a.nweight2_code1 == b.nweight2_code1 &&
			a.nweight2_code2 == b.nweight2_code2;
	}

	template <class W>
	inline std::size_t hash_value(const sum_key<W>& key) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.id_1);
		boost::hash_combine(seed, key.id_2);
		for (const auto& code : key.nweight1_code1) {
			boost::hash_combine(seed, code);
		}
		for (const auto& code : key.nweight1_code2) {
			boost::hash_combine(seed, code);
		}
		for (const auto& code : key.nweight2_code1) {
			boost::hash_combine(seed, code);
		}
		for (const auto& code : key.nweight2_code2) {
			boost::hash_combine(seed, code);
		}
		return seed;
	}

	template <class W>
	using sum_table = boost::unordered_map<sum_key<W>, node::weightednode<W>>;


	typedef std::vector<std::pair<int, int>> pair_cmd;

	// the type for trace cache
	template <class W>
	struct trace_key {
		int id;
		// first: the smaller index to trace, second: the larger index to trace
		pair_cmd remained_ls;
		// first: the larger index to trace, seconde; the index value to select
		pair_cmd waiting_ls;

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
		inline trace_key(int _id, const pair_cmd& _remained_ls, const pair_cmd& _waiting_ls) {
			id = _id;
			remained_ls = pair_cmd(_remained_ls);
			waiting_ls = pair_cmd(_waiting_ls);
		}

		inline trace_key(int _id, pair_cmd&& _remained_ls, pair_cmd&& _waiting_ls) {
			id = _id;
			remained_ls = std::move(_remained_ls);
			waiting_ls = std::move(_waiting_ls);
		}

		inline trace_key(const trace_key& other) {
			id = other.id;
			remained_ls = other.remained_ls;
			waiting_ls = other.waiting_ls;
		}
		inline trace_key& operator =(trace_key&& other) {
			id = other.id;
			remained_ls = std::move(other.remained_ls);
			waiting_ls = std::move(other.waiting_ls);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const trace_key<W>& a, const trace_key<W>& b) {
		return (a.id == b.id && a.remained_ls == b.remained_ls && a.waiting_ls == b.waiting_ls);
	}

	template <class W>
	inline std::size_t hash_value(const trace_key<W>& key) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.id);
		for (const auto& cmd : key.remained_ls) {
			boost::hash_combine(seed, cmd.first);
			boost::hash_combine(seed, cmd.second);
		}
		for (const auto& cmd : key.waiting_ls) {
			boost::hash_combine(seed, cmd.first);
			boost::hash_combine(seed, cmd.second);
		}
		return seed;
	}

	template <class W>
	using trace_table = boost::unordered_map<trace_key<W>, node::weightednode<W>>;

	
	// the type for contract cache
	// under extreame case this key is not secure, because the "data shape" information is not stored here.
	// It is a potential flaw, but we can avoid it by using it only in quantum circuits.
	// It is left for future, when quantum situation is devided from general situation.
	template <class W>
	struct cont_key {
		const node::Node<W>* p_a;
		const node::Node<W>* p_b;
		// first: the smaller index to trace, second: the larger index to trace
		pair_cmd remained_ls;
		// first: the larger index to trace, seconde; the index value to select
		pair_cmd a_waiting_ls;
		pair_cmd b_waiting_ls;
		// true: the next index is from A. false: the next index is from B.
		std::vector<int64_t> a_new_order;
		std::vector<int64_t> b_new_order;

		inline cont_key(const node::Node<W>* _p_a, const node::Node<W>* _p_b, const pair_cmd& _remained_ls, 
			const pair_cmd& _a_waiting_ls, const pair_cmd& _b_waiting_ls,
			const std::vector<int64_t>& _a_new_order, int pos_a, const std::vector<int64_t>& _b_new_order, int pos_b) {

			// in theory, we can further increase the reuseage of cont_cache, by eliminating the swaping freedom
			// but it is time consuming and maybe not that worthy.
			p_a = _p_a;
			p_b = _p_b;
			remained_ls = _remained_ls;
			a_waiting_ls = _a_waiting_ls;
			b_waiting_ls = _b_waiting_ls;
			a_new_order = std::vector<int64_t>(_a_new_order.begin() + pos_a, _a_new_order.end());
			b_new_order = std::vector<int64_t>(_b_new_order.begin() + pos_b, _b_new_order.end());
		}

		inline cont_key(const node::Node<W>* _p_a, const node::Node<W>* _p_b, pair_cmd&& _remained_ls, 
			pair_cmd&& _a_waiting_ls, pair_cmd&& _b_waiting_ls,
			const std::vector<int64_t>& _a_new_order, int pos_a, const std::vector<int64_t>& _b_new_order, int pos_b) {
			p_a = _p_a;
			p_b = _p_b;
			remained_ls = std::move(_remained_ls);
			a_waiting_ls = std::move(_a_waiting_ls);
			b_waiting_ls = std::move(_b_waiting_ls);
			a_new_order = std::vector<int64_t>(_a_new_order.begin() + pos_a, _a_new_order.end());
			b_new_order = std::vector<int64_t>(_b_new_order.begin() + pos_b, _b_new_order.end());
		}

		inline cont_key(const cont_key& other) {
			p_a = other.p_a;
			p_b = other.p_b;
			remained_ls = other.remained_ls;
			a_waiting_ls = other.a_waiting_ls;
			b_waiting_ls = other.b_waiting_ls;
			a_new_order = other.a_new_order;
			b_new_order = other.b_new_order;
		}
		inline cont_key& operator =(cont_key&& other) {
			p_a = other.p_a;
			p_b = other.p_b;
			remained_ls = std::move(other.remained_ls);
			a_waiting_ls = std::move(other.a_waiting_ls);
			b_waiting_ls = std::move(other.b_waiting_ls);
			a_new_order = std::move(other.a_new_order);
			b_new_order = std::move(other.b_new_order);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const cont_key<W>& a, const cont_key<W>& b) {
		return (a.p_a == b.p_a && a.p_b == b.p_b &&
			a.remained_ls == b.remained_ls && a.a_waiting_ls == b.a_waiting_ls && a.b_waiting_ls == b.b_waiting_ls &&
			a.a_new_order == b.a_new_order && a.b_new_order == b.b_new_order);
	}

	template <class W>
	inline std::size_t hash_value(const cont_key<W>& key) {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.p_a);
		boost::hash_combine(seed, key.p_b);
		for (const auto& cmd : key.remained_ls) {
			boost::hash_combine(seed, cmd.first);
			boost::hash_combine(seed, cmd.second);
		}
		for (const auto& cmd : key.a_waiting_ls) {
			boost::hash_combine(seed, cmd.first);
			boost::hash_combine(seed, cmd.second);
		}
		for (const auto& cmd : key.b_waiting_ls) {
			boost::hash_combine(seed, cmd.first);
			boost::hash_combine(seed, cmd.second);
		}
		for (const auto& order_a : key.a_new_order) {
			boost::hash_combine(seed, order_a);
		}
		for (const auto& order_b : key.b_new_order) {
			boost::hash_combine(seed, order_b);
		}
		return seed;
	}

	template <class W>
	using cont_table = boost::unordered_map<cont_key<W>, node::weightednode<W>>;




	template <class W>
	struct Global_Cache {
		static duplicate_table<W>* p_duplicate_cache;
		static append_table<W>* p_append_cache;
		static CUDAcpl_table<W>* p_CUDAcpl_cache;
		static sum_table<W>* p_sum_cache;
		static trace_table<W>* p_trace_cache;
		static cont_table<W>* p_cont_cache;
	};
}