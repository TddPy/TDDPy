#pragma once
#include "stdafx.h"
#include "weight.hpp"

namespace node {
	template <class W>
	class Node;
	template <class W>
	struct weightednode;
	template <class W>
	struct wnode_cache;
	template <class W>
	using succ_ls = std::vector<weightednode<W>>;
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
		unique_table_key(int _order, const node::succ_ls<W>& successors) noexcept {
			if constexpr (std::is_same_v<W, wcomplex>) {
				order = _order;
				code1 = std::vector<weight::WCode>(successors.size());
				code2 = std::vector<weight::WCode>(successors.size());
				nodes = std::vector<const node::Node<wcomplex>*>(successors.size());
				for (int i = 0; i < successors.size(); i++) {
					weight::get_int_key(code1.data() + i, successors[i].weight.real());
					weight::get_int_key(code2.data() + i, successors[i].weight.imag());
					nodes[i] = successors[i].get_node();
				}
			}
			else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
				order = _order;
				auto numel = successors[0].weight.numel();
				code1 = std::vector<weight::WCode>(numel * successors.size());
				nodes = std::vector<const node::Node<CUDAcpl::Tensor>*>(successors.size());
				for (int i = 0; i < successors.size(); i++) {
					weight::get_int_key(code1.data() + i * numel, successors[i].weight);
					nodes[i] = successors[i].get_node();
				}
				code2 = std::vector<weight::WCode>(successors[0].weight.dim() - 1);
				auto&& sizes = successors[0].weight.sizes();
				for (int i = 0; i < successors[0].weight.dim() - 1; i++) {
					code2[i] = sizes[i];
				}
			}
		}

		unique_table_key(const unique_table_key& other) noexcept {
			order = other.order;
			code1 = other.code1;
			code2 = other.code2;
			nodes = other.nodes;
		}

		unique_table_key& operator =(unique_table_key&& other) noexcept {
			order = other.order;
			code1 = std::move(other.code1);
			code2 = std::move(other.code2);
			nodes = std::move(other.nodes);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const unique_table_key<W>& a, const unique_table_key<W>& b) noexcept {
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
	inline std::size_t hash_value(const unique_table_key<W>& key) noexcept {
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
	using unique_table = boost::unordered_map<unique_table_key<W>, node::Node<W>*>;



	template <class W>
	struct CUDAcpl_table_key {
		const node::Node<W>* p_node;
		std::vector<int64_t> inner_shape;

		CUDAcpl_table_key() {}

		/// <summary>
		/// Construction of a unique_table key (complex version)
		/// </summary>
		/// <param name="_order"></param>
		/// <param name="_range"></param>
		/// <param name="_p_weights">[borrowed]</param>
		/// <param name="_p_nodes">[borrowed</param>
		CUDAcpl_table_key(const node::Node<W>* _p_node, const std::vector<int64_t>& _inner_shape) noexcept {
			p_node = _p_node;
			inner_shape = _inner_shape;
		}

		CUDAcpl_table_key(const CUDAcpl_table_key& other) noexcept {
			p_node = other.p_node;
			inner_shape = other.inner_shape;
		}

		CUDAcpl_table_key& operator =(CUDAcpl_table_key&& other) noexcept {
			p_node = other.p_node;
			inner_shape = std::move(other.inner_shape);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const CUDAcpl_table_key<W>& a, const CUDAcpl_table_key<W>& b) noexcept {
		return a.p_node == b.p_node && a.inner_shape == b.inner_shape;
	}

	template <class W>
	inline std::size_t hash_value(const CUDAcpl_table_key<W>& key) noexcept {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.p_node);
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
		const node::Node<W>* p_node_1;
		std::vector<weight::WCode> nweight1_code1;
		std::vector<weight::WCode> nweight1_code2;
		const node::Node<W>* p_node_2;
		std::vector<weight::WCode> nweight2_code1;
		std::vector<weight::WCode> nweight2_code2;

		//sum_key() {};
		
		/// <summary>
		/// Construct the key. Note that id_1 will be set as the smaller one.
		/// </summary>
		/// <param name="id_a"></param>
		/// <param name="weight_a"></param>
		/// <param name="id_b"></param>
		/// <param name="weight_b"></param>
		sum_key(const node::Node<W>* p_node_a, const W& weight_a, const node::Node<W>* p_node_b, const W& weight_b) {
			if constexpr (std::is_same_v<W, wcomplex>) {
				if (p_node_a < p_node_b) {
					p_node_1 = p_node_a;
					p_node_2 = p_node_b;
					nweight1_code1 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight1_code1.data(), weight_a.real());
					nweight1_code2 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight1_code2.data(), weight_a.imag());
					nweight2_code1 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight2_code1.data(), weight_b.real());
					nweight2_code2 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight2_code2.data(), weight_b.imag());
				}
				else {
					p_node_1 = p_node_b;
					p_node_2 = p_node_a;
					nweight1_code1 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight1_code1.data(), weight_b.real());
					nweight1_code2 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight1_code2.data(), weight_b.imag());
					nweight2_code1 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight2_code1.data(), weight_a.real());
					nweight2_code2 = std::vector<weight::WCode>(1);
					weight::get_int_key(nweight2_code2.data(), weight_a.imag());
				}
			}
			else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
				if (p_node_a < p_node_b) {
					p_node_1 = p_node_a;
					p_node_2 = p_node_b;
					nweight1_code1 = std::vector<weight::WCode>(weight_a.numel());
					weight::get_int_key(nweight1_code1.data(), weight_a);

					// store the shape
					nweight1_code2 = std::vector<weight::WCode>(weight_a.dim() - 1);
					auto&& sizes = weight_a.sizes();
					for (int i = 0; i < weight_a.dim() - 1; i++) {
						nweight1_code2[i] = sizes[i];
					}

					nweight2_code1 = std::vector<weight::WCode>(weight_b.numel());
					weight::get_int_key(nweight2_code1.data(), weight_b);

					// nweight2_code2 is not needed
					nweight2_code2 = std::vector<weight::WCode>();
				}
				else {
					p_node_1 = p_node_b;
					p_node_2 = p_node_a;
					nweight1_code1 = std::vector<weight::WCode>(weight_b.numel());
					weight::get_int_key(nweight1_code1.data(), weight_b);

					// store the shape
					nweight1_code2 = std::vector<weight::WCode>(weight_b.dim() - 1);
					auto&& sizes = weight_b.sizes();
					for (int i = 0; i < weight_b.dim() - 1; i++) {
						nweight1_code2[i] = sizes[i];
					}

					nweight2_code1 = std::vector<weight::WCode>(weight_a.numel());
					weight::get_int_key(nweight2_code1.data(), weight_a);

					nweight2_code2 = std::vector<weight::WCode>();
				}
			}
		}
		
		sum_key(const sum_key& other) noexcept {
			p_node_1 = other.p_node_1;
			nweight1_code1 = other.nweight1_code1;
			nweight1_code2 = other.nweight1_code2;
			p_node_2 = other.p_node_2;
			nweight2_code1 = other.nweight2_code1;
			nweight2_code2 = other.nweight2_code2;
		}

		sum_key& operator =(sum_key&& other) noexcept {
			p_node_1 = other.p_node_1;
			nweight1_code1 = std::move(other.nweight1_code1);
			nweight1_code2 = std::move(other.nweight1_code2);
			p_node_2 = other.p_node_2;
			nweight2_code1 = std::move(other.nweight2_code1);
			nweight2_code2 = std::move(other.nweight2_code2);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const sum_key<W>& a, const sum_key<W>& b) noexcept {
		return a.p_node_1 == b.p_node_1 && a.p_node_2 == b.p_node_2 &&
			a.nweight1_code1 == b.nweight1_code1 &&
			a.nweight1_code2 == b.nweight1_code2 &&
			a.nweight2_code1 == b.nweight2_code1 &&
			a.nweight2_code2 == b.nweight2_code2;
	}

	template <class W>
	inline std::size_t hash_value(const sum_key<W>& key) noexcept {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.p_node_1);
		boost::hash_combine(seed, key.p_node_2);
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
	using sum_table = boost::unordered_map<sum_key<W>, node::wnode_cache<W>>;


	typedef std::vector<std::pair<int, int>> pair_cmd;

	// the type for trace cache
	template <class W>
	struct trace_key {
		const node::Node<W>* p_node;
		// first: the smaller index to trace, second: the larger index to trace
		pair_cmd remained_ls;
		// first: the larger index to trace, seconde; the index value to select
		pair_cmd waiting_ls;

		//trace_key() {};

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
		inline trace_key(const node::Node<W>* _p_node, const pair_cmd& _remained_ls, const pair_cmd& _waiting_ls) noexcept {
			p_node = _p_node;
			remained_ls = pair_cmd(_remained_ls);
			waiting_ls = pair_cmd(_waiting_ls);
		}

		inline trace_key(const node::Node<W>* _p_node, pair_cmd&& _remained_ls, pair_cmd&& _waiting_ls) noexcept {
			p_node = _p_node;
			remained_ls = std::move(_remained_ls);
			waiting_ls = std::move(_waiting_ls);
		}

		inline trace_key(const trace_key& other) noexcept {
			p_node = other.p_node;
			remained_ls = other.remained_ls;
			waiting_ls = other.waiting_ls;
		}
		inline trace_key& operator =(trace_key&& other) noexcept {
			p_node = other.p_node;
			remained_ls = std::move(other.remained_ls);
			waiting_ls = std::move(other.waiting_ls);
			return *this;
		}
	};

	template <class W>
	inline bool operator == (const trace_key<W>& a, const trace_key<W>& b) noexcept {
		return (a.p_node == b.p_node && a.remained_ls == b.remained_ls && a.waiting_ls == b.waiting_ls);
	}

	template <class W>
	inline std::size_t hash_value(const trace_key<W>& key) noexcept {
		std::size_t seed = 0;
		boost::hash_combine(seed, key.p_node);
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
	using trace_table = boost::unordered_map<trace_key<W>, node::wnode_cache<W>>;

	
	// the type for contract cache
	// under extreame case this key is not secure, because the "data shape" information is not stored here.
	// It is a potential flaw, but we can avoid it by using it only in quantum circuits.
	// It is left for future, when quantum situation is devided from general situation.
	template <class W1,class W2>
	struct cont_key {
		const node::Node<W1>* p_a;
		const node::Node<W2>* p_b;
		// first: the smaller index to trace, second: the larger index to trace
		pair_cmd remained_ls;
		// first: the larger index to trace, seconde; the index value to select
		pair_cmd a_waiting_ls;
		pair_cmd b_waiting_ls;
		// true: the next index is from A. false: the next index is from B.
		std::vector<int64_t> a_new_order;
		std::vector<int64_t> b_new_order;

		bool parallel_tensor;

		//cont_key() {};

		inline cont_key(const node::Node<W1>* _p_a, const node::Node<W2>* _p_b, const pair_cmd& _remained_ls, 
			const pair_cmd& _a_waiting_ls, const pair_cmd& _b_waiting_ls,
			const std::vector<int64_t>& _a_new_order, int pos_a, const std::vector<int64_t>& _b_new_order, int pos_b, bool _parallel_tensor) {

			// in theory, we can further increase the reuseage of cont_cache, by eliminating the swaping freedom
			// but it is time consuming and maybe not that worthy.
			p_a = _p_a;
			p_b = _p_b;
			remained_ls = _remained_ls;
			a_waiting_ls = _a_waiting_ls;
			b_waiting_ls = _b_waiting_ls;
			a_new_order = std::vector<int64_t>(_a_new_order.begin() + pos_a, _a_new_order.end());
			b_new_order = std::vector<int64_t>(_b_new_order.begin() + pos_b, _b_new_order.end());
			parallel_tensor = _parallel_tensor;
		}

		inline cont_key(const node::Node<W1>* _p_a, const node::Node<W2>* _p_b, pair_cmd&& _remained_ls, 
			pair_cmd&& _a_waiting_ls, pair_cmd&& _b_waiting_ls,
			const std::vector<int64_t>& _a_new_order, int pos_a, const std::vector<int64_t>& _b_new_order, int pos_b, bool _parallel_tensor) noexcept {
			p_a = _p_a;
			p_b = _p_b;
			remained_ls = std::move(_remained_ls);
			a_waiting_ls = std::move(_a_waiting_ls);
			b_waiting_ls = std::move(_b_waiting_ls);
			a_new_order = std::vector<int64_t>(_a_new_order.begin() + pos_a, _a_new_order.end());
			b_new_order = std::vector<int64_t>(_b_new_order.begin() + pos_b, _b_new_order.end());
			parallel_tensor = _parallel_tensor;
		}

		inline cont_key(const cont_key& other) noexcept {
			p_a = other.p_a;
			p_b = other.p_b;
			remained_ls = other.remained_ls;
			a_waiting_ls = other.a_waiting_ls;
			b_waiting_ls = other.b_waiting_ls;
			a_new_order = other.a_new_order;
			b_new_order = other.b_new_order;
			parallel_tensor = other.parallel_tensor;
		}
		inline cont_key& operator =(cont_key&& other) noexcept {
			p_a = other.p_a;
			p_b = other.p_b;
			remained_ls = std::move(other.remained_ls);
			a_waiting_ls = std::move(other.a_waiting_ls);
			b_waiting_ls = std::move(other.b_waiting_ls);
			a_new_order = std::move(other.a_new_order);
			b_new_order = std::move(other.b_new_order);
			parallel_tensor = std::move(other.parallel_tensor);
			return *this;
		}
	};

	template <class W1, class W2>
	inline bool operator == (const cont_key<W1, W2>& a, const cont_key<W1, W2>& b) noexcept {
		return (a.p_a == b.p_a && a.p_b == b.p_b &&
			a.remained_ls == b.remained_ls && a.a_waiting_ls == b.a_waiting_ls && a.b_waiting_ls == b.b_waiting_ls &&
			a.a_new_order == b.a_new_order && a.b_new_order == b.b_new_order && a.parallel_tensor == b.parallel_tensor);
	}

	template <class W1, class W2>
	inline std::size_t hash_value(const cont_key<W1, W2>& key) noexcept {
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
		boost::hash_combine(seed, key.parallel_tensor);
		return seed;
	}


	template <typename W1, typename W2>
	using cont_table = boost::unordered_map<cont_key<W1, W2>, node::wnode_cache<weight::W_C<W1, W2>>>;


	template <typename CACHE>
	inline void clean_garbage(std::pair<std::shared_mutex, CACHE>& the_cache) {
		the_cache.first.lock();
		for (auto&& i = the_cache.second.begin(); i != the_cache.second.end();) {
			if (i->second.is_garbage()) {
				i = the_cache.second.erase(i);
			}
			else {
				i++;
			}
		}
		the_cache.first.unlock();
	}

	template <typename W>
	struct Global_Cache {
		static std::pair<std::shared_mutex, CUDAcpl_table<W>> CUDAcpl_cache;
		static std::pair<std::shared_mutex, sum_table<W>> sum_cache;
		static std::pair<std::shared_mutex, trace_table<W>> trace_cache;
	};

	template <typename W1, typename W2>
	struct Cont_Cache {
		static std::pair<std::shared_mutex, cont_table<W1, W2>> cont_cache;
	};
}