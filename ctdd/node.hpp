#pragma once
#include "stdafx.h"
#include "cache.hpp"
#include "succ_ls.hpp"

namespace node {

	// The node used in tdd.
	template <class W>
	class Node {
	private:
		/* record the size of m_unique_table
		*  Note: id = 0 is reserved for terminal node (null).
		*/
		static int m_global_id;

		// The unique_table to store all the node instances used in tdd.
		static cache::unique_table<W> m_unique_table;

		int m_id;

		//represent the order of this node (which tensor index it represent)
		int m_order;

		/* The weight and node of the successors
		*  Note: terminal nodes are represented by nullptr in the successors.
		*/
		succ_ls<W> m_successors;

	private:


		/// <summary>
		/// Count all the nodes starting from this node.
		/// </summary>
		/// <param name="id_ls"> the vector to store all the ids</param>
		void node_search(std::vector<int>& id_ls) const {
			// check whether it is in p_id already
			if (std::find(id_ls.cbegin(), id_ls.cend(), m_id) != id_ls.cend()) {
				return;
			}
			// it is not counted yet in this case
			id_ls.push_back(m_id);
			for (const auto& succ : m_successors) {
				if (!succ.isterminal()) {
					succ.node->node_search(id_ls);
				}
			}
		}

		static const Node<W>* duplicate_iterate(const node::Node<W>* p_node, int order_shift,
			const std::vector<int64_t>& parallel_shape,
			const std::vector<int64_t>& shape_front,
			const std::vector<int64_t>& shape_back) {
			if (p_node == nullptr) {
				return nullptr;
			}

			auto&& order = p_node->m_order + order_shift;

			// p_node is guaranteed not to be null
			auto&& key = std::make_pair(p_node->m_id, order_shift);
			auto&& p_find_res = cache::Global_Cache<W>::p_duplicate_cache->find(key);
			if (p_find_res != cache::Global_Cache<W>::p_duplicate_cache->end()) {
				return p_find_res->second;
			}
			else {
				//broadcast the dims for tensor weight
				//...

				auto&& new_successors = succ_ls<W>(p_node->m_successors);
				for (auto&& succ : new_successors) {
					succ.node = Node::duplicate_iterate(succ.node, order_shift, parallel_shape, shape_front, shape_back);
				}
				auto&& p_res = Node<W>::get_unique_node(order, std::move(new_successors));
				cache::Global_Cache<W>::p_duplicate_cache->insert(std::make_pair(std::move(key), p_res));
				return p_res;
			}
		}


		static const Node<W>* shift_multiple_iterate(const Node<W>* p_node, const std::vector<int64_t>& new_order_ls,
			boost::unordered_map<int, const Node<W>*>& local_shift_cache) {
			if (p_node == nullptr) {
				return nullptr;
			}

			// p_node is guaranteed not to be null
			auto&& order = new_order_ls[p_node->m_order];
			auto&& key = p_node->m_id;
			auto&& p_find_res = local_shift_cache.find(key);
			if (p_find_res != local_shift_cache.end()) {
				return p_find_res->second;
			}
			else {
				auto&& new_successors = succ_ls<W>(p_node->m_successors);
				for (auto&& succ : new_successors) {
					succ.node = shift_multiple_iterate(succ.node, new_order_ls, local_shift_cache);
				}
				auto&& p_res = Node<W>::get_unique_node(order, new_successors);
				local_shift_cache.insert(std::make_pair(std::move(key), p_res));
				return p_res;
			}
		}

		static const Node<W>* append_iterate(const Node<W>* p_nodea, const Node<W>* p_nodeb) {
			if (p_nodea == nullptr) {
				return p_nodeb;
			}

			auto&& key = cache::append_table_key<W>(p_nodea->m_id, node::Node<W>::get_id_all(p_nodeb));
			auto&& p_res_find = cache::Global_Cache<W>::p_append_cache->find(key);
			if (p_res_find != cache::Global_Cache<W>::p_append_cache->end()) {
				return p_res_find->second;
			}
			else {
				auto&& new_successors = succ_ls<W>(p_nodea->m_successors);
				for (auto&& succ : new_successors) {
					if (weight::is_zero(succ.weight)) {
						// redirect zero weight to terminal nodes
						succ.node = nullptr;
					}
					else {
						succ.node = append_iterate(succ.node, p_nodeb);
					}
				}
				auto&& p_res = Node<W>::get_unique_node(p_nodea->m_order, std::move(new_successors));
				cache::Global_Cache<W>::p_append_cache->insert(std::make_pair(std::move(key), p_res));
				return p_res;
			}
		}

	public:

		inline static const cache::unique_table<W>& get_unique_table() {
			return m_unique_table;
		}

		/// <summary>
		/// Note: the memory of successors will be transfered to this node, therefore a right value is needed.
		/// </summary>
		/// <param name="order"></param>
		/// <param name="successors"></param>
		Node(int order, succ_ls<W>&& successors) {
			m_global_id += 1;
			m_id = m_global_id;
			m_order = order;
			m_successors = std::move(successors);
		}

		static void reset() {
			m_global_id = 0;
			for (auto&& i : m_unique_table) {
				delete i.second;
			}
			m_unique_table.clear();
		}

		/// <summary>
		/// Note: when the successors passed in is a left value, it will be copied first.
		/// When the equality checking inside is conducted with the node.EPS tolerance.So feel free
		/// to pass in the raw weights from calculation.
		/// </summary>
		/// <param name="order"></param>
		/// <param name="successors"></param>
		/// <returns></returns>
		static const Node<W>* get_unique_node(int order, const succ_ls<W>& successors) {
			auto&& key = cache::unique_table_key<W>(order, successors);
			auto&& p_find_res = m_unique_table.find(key);
			if (p_find_res != m_unique_table.end()) {
				return p_find_res->second;
			}

			node::Node<W>* p_node = new node::Node<W>(order, succ_ls<W>(successors));
			m_unique_table.insert(std::make_pair(std::move(key), p_node));
			return p_node;
		}

		/// <summary>
		/// Note: when the successors passed in is a right value, it will be transfered to the new node.
		/// When the equality checking inside is conducted with the node.EPS tolerance.So feel free
		/// to pass in the raw weights from calculation.
		/// </summary>
		/// <param name="order"></param>
		/// <param name="successors"></param>
		/// <returns></returns>
		static const Node<W>* get_unique_node(int order, succ_ls<W>&& successors) {
			auto&& key = cache::unique_table_key<W>(order, successors);
			auto&& p_find_res = m_unique_table.find(key);
			if (p_find_res != m_unique_table.end()) {
				return p_find_res->second;
			}

			node::Node<W>* p_node = new node::Node<W>(order, std::move(successors));
			m_unique_table.insert(std::make_pair(std::move(key), p_node));
			return p_node;
		}

		inline static int get_id_all(const Node* p_node) {
			if (p_node == nullptr) {
				return 0;
			}
			return p_node->m_id;
		}

		inline int get_id() const {
			return m_id;
		}

		inline int get_order() const {
			return m_order;
		}

		inline int get_range() const {
			return m_successors.size();
		}

		void print() const {
			for (int i = 0; i < m_order; i++) {
				std::cout << "-";
			}
			std::cout << "=======" << std::endl;
			for (int i = 0; i < m_order; i++) {
				std::cout << " ";
			}
			std::cout << "|node: " << this << std::endl;

			for (int i = 0; i < m_order; i++) {
				std::cout << " ";
			}
			std::cout << "|id: " << m_id << std::endl;

			for (int i = 0; i < m_order; i++) {
				std::cout << " ";
			}
			std::cout << "|order: " << m_order << std::endl;

			for (int i = 0; i < m_order; i++) {
				std::cout << " ";
			}
			std::cout << "|successors: " << std::endl;

			for (int j = 0; j < m_successors.size(); j++) {
				for (int i = 0; i < m_order; i++) {
					std::cout << " ";
				}
				std::cout << "|  " << j << " " << "weight: " << m_successors[j].weight << std::endl;
				for (int i = 0; i < m_order; i++) {
					std::cout << " ";
				}
				std::cout << "|  " << j << " " << "node: " << m_successors[j].node << std::endl;
			}

			for (const auto& p : m_successors) {
				if (p->node != nullptr) {
					p->node->print();
				}
			}
		}

		inline const succ_ls<W>& get_successors() const {
			return m_successors;
		}

		/// <summary>
		/// Count all the nodes starting from this one.
		/// </summary>
		/// <returns></returns>
		inline int get_size() const {
			auto&& id_ls = std::vector<int>();
			node_search(id_ls);
			return id_ls.size();
		}

		/// <summary>
		/// Get the corresponding key structure of this node.
		/// </summary>
		/// <returns></returns>
		inline cache::unique_table_key<W> get_key_struct() const {
			return unique_table_key(m_order, m_successors);
		}

		/// <summary>
		/// Calculate and return the Hash value of the node (can be nullptr).
		/// </summary>
		/// <param name="p_node"></param>
		/// <returns></returns>
		inline static std::size_t get_hash(const Node<W>* p_node) {
			if (p_node == nullptr) {
				return 0;
			}
			std::size_t seed = 0;
			return cache::hash_value(p_node->get_key_struct());
		}

		/// <summary>
		/// Duplicate from this node, with the initial order of (node.order + order_shift),
		///	and broadcast it to contain the extra(parallel index) shape aheadand behind.
		/// </summary>
		/// <param name="p_node"></param>
		/// <param name="order_shift"></param>
		/// <returns></returns>
		inline static const Node<W>* duplicate(const Node* p_node, int order_shift,
			const std::vector<int64_t>& parallel_shape,
			const std::vector<int64_t>& shape_front,
			const std::vector<int64_t>& shape_back) {
			return duplicate_iterate(p_node, order_shift, parallel_shape, shape_front, shape_back);
		}

		/// <summary>
		/// Shift the order of node, Return the result.
		/// order of new node is p_new_order[node.order]
		/// </summary>
		/// <returns></returns>
		inline static const Node<W>* shift_multiple(const Node<W>* p_node, const std::vector<int64_t>& new_order_ls) {

			// note that the local cache need the 'node.id' as key only, so as to balance the key structure complexity.
			// however, therefore this cache only works in one invocation
			auto&& local_shift_cache = boost::unordered_map<int, const Node<W>*>();
			return shift_multiple_iterate(p_node, new_order_ls, local_shift_cache);
		}


		/// <summary>
		/// Replace the terminal node in this graph with 'node', and return the result.
		/// depth: 
		/// parallel_tensor : whether to tensor on the parallel indices
		/// Note : it should be considered merely as an operation on node structures, with no meaning in the tensor regime.
		/// </summary>
		/// <param name="p_a"></param>
		/// <param name="a_depth">the depth from 'node a' on, i.e.the number of dims corresponding to this node.</param>
		/// <param name="p_b"></param>
		/// <param name="parallel_tensor"></param>
		/// <returns></returns>
		static const Node<W>* append(const Node<W>* p_a, int a_depth, const Node<W>* p_b,
			const std::vector<int64_t>& parallel_shape,
			const std::vector<int64_t>& shape_front,
			const std::vector<int64_t>& shape_back,
			bool parallel_tensor = false) {
			if (!parallel_tensor) {
				auto&& modified_node = node::Node<W>::duplicate(p_b, a_depth, parallel_shape, shape_front, shape_back);
				return node::Node<W>::append_iterate(p_a, modified_node);
			}
			else {
				//not implemented yet
				throw - 1;
			}
		}
	};
}