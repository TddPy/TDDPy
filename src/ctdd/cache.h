#pragma once


#include <boost/unordered_map.hpp>
#include "weight.hpp"

namespace node {
	template <typename T_WEIGHT>
	class Node;
	template <typename T_WEIGHT>
	struct WNode;

	template <typename T_WEIGHT>
	using succ_ls = std::vector<WNode<T_WEIGHT>>;
}


namespace cache {



	// The key that uniquely determine a node in the diagram.
	// The key is composed by the order of the node and the successors.
	template <typename T_WCODE, class T_WEIGHT>
	struct unique_table_key {
		int order;
		succ_ls<T_WEIGHT> succ;	// We need to use [vector<WNode>], not [vector<WNode*>], because the key should work independently.
	}
	// TODO : due to normalization, the first element in codes may be sometimes unnecessary?

	template <class W>
	inline std::size_t hash_value(const succ_ls<W>& key) noexcept {
		std::size_t seed = 0;
		for (const auto& item : key) {
			boost::hash_combine(seed, key);
		}
		return seed;
	}

	/*	May be we can come up with a better hash function!!!!!!!!!!!!!!!!!!!!!!!!!!!
	*/


	template <typename W>
	using unique_table = boost::unordered_map<succ_ls<W>, node::Node<W>*>;






	// the type for element_table cache

	// match nodes to elements
	template <typename W>
	inline std::size_t hash_value(node::Node<W>*& key) noexcept {
		return key;	// the memory address should serve as a good hash value
	}
	
	template <class W>
	using element_table = boost::unordered_map<node::Node<W>*, CUDAcpl::Tensor>;


	// /// <summary>
	// /// the type for summation cache
	// /// </summary>
	// /// <typeparam name="W"></typeparam>
	// template <class  W>
	// struct sum_key {
	// 	const node::wnode<W>* p_wnode_1;
	// 	const node::wnode<W>* p_wnode_2;

	// 	sum_key(const node::wnode<W>* _p_wnode_1, const node::wnode<W>*_ p_wnode_2) {
	// 		p_wnode_1 = _p_wnode_1;
	// 		p_wnode_2 = _p_wnode_2;
	// 	}

	// 	// inline bool is_garbage() const noexcept {
	// 	// 	return node::Node<W>::is_garbage(p_node_1) || node::Node<W>::is_garbage(p_node_2);
	// 	// }
	// };

	// template <class W>
	// inline bool operator == (const sum_key<W>& a, const sum_key<W>& b) noexcept {
	// 	return *a.p_wnode_1 == *b.p_wnode_1 && *a.p_wnode_2 == *b.p_wnode_2;
	// }

	// template <class W>
	// inline std::size_t hash_value(const sum_key<W>& key) noexcept {
	// 	throw -1;
	// }

	// template <class W>
	// using sum_table = boost::unordered_map<sum_key<W>, node::wnode_cache<W>>;

}