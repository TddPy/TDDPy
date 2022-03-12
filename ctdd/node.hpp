#pragma once
#include "stdafx.h"
#include "cache.hpp"
#include "succ_ls.hpp"

namespace node {

	// The node used in tdd.
	template <typename W>
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

			for (const auto& succ : m_successors) {
				if (succ.node != nullptr) {
					succ.node->print();
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
			// the terminal node is counted
			return id_ls.size() + 1;
		}
	};
}