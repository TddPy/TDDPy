#pragma once

#include <vector>

#include "typing.hpp"
#include "cache.hpp"



namespace node {

	template <class T_WEIGHT>
	using succ_ls = std::vector<wnode::WNode<T_WEIGHT>>;


	// The node used in tdd.
	template <typename T_WEIGHT>
	class Node {
	private:

		// The unique_table to store all the node instances used in tdd.
		static cache::unique_table<T_WEIGHT> m_unique_table;

		// represent the order of this node (which tensor index it represent)
		const int m_order;

		/* The weight and node of the successors
		*  Note: terminal nodes are represented by nullptr in the successors.
		*/
		succ_ls<T_WEIGHT> m_successors;


	public:

		template <typename T_WEIGHT>
		friend class WNode;

		inline int get_order() const noexcept {
			return m_order;
		}

        // get the number of possible values for the index of this node
		inline int get_range() const noexcept {
			return m_successors.size();
		}

		inline const succ_ls<T_WEIGHT>& get_successors() const noexcept {
			return m_successors;
		}

		inline static const cache::unique_table<T_WEIGHT> get_unique_table() {
			return m_unique_table;
		}
	};
}