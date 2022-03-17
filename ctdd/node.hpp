#pragma once
#include "stdafx.h"
#include "cache.hpp"

namespace node {

	template <class W>
	using succ_ls = std::vector<weightednode<W>>;


	// The node used in tdd.
	template <typename W>
	class Node {
	private:

		// The unique_table to store all the node instances used in tdd.
		static cache::unique_table<W>* mp_unique_table;
		static std::shared_mutex unique_table_m;

		// represent the order of this node (which tensor index it represent)
		const int m_order;

		// reference count : the number of incoming edges of this node, in the whole net
		// principles for reference count management: 
		//  0. reference count is triggered by weightednode construction and deconstruction
		//	1. who owns the object is responsible for decrease the reference count when deconstructing
		//	2. if std::move is used, than the ownership is transfered 
		std::atomic_int m_ref_count;

		/* The weight and node of the successors
		*  Note: terminal nodes are represented by nullptr in the successors.
		*/
		const succ_ls<W> m_successors;

	private:


		/// <summary>
		/// Count all the nodes starting from this node.
		/// </summary>
		/// <param name="id_ls"> the vector to store all the ids</param>
		void node_search(boost::unordered_set<const Node<W>*>& node_ls) const {
			// check whether it is in node_ls already, and insert in
			auto&& insert_res = node_ls.insert(this);

			if (insert_res.second) {
				// it is not counted yet in this case
				for (const auto& succ : m_successors) {
					if (!succ.isterminal()) {
						succ.get_node()->node_search(node_ls);
					}
				}
			}
		}


	public:

		template <typename W>
		friend class weightednode;

		// note that due to the transfer semantics of successors, their reference counts will not be increased.
		Node(int order, succ_ls<W>&& successors) :m_order(order), m_ref_count(1), m_successors(std::move(successors)) { }

		inline int get_ref_count() const {
			return m_ref_count.load(std::memory_order::memory_order_relaxed);
		}

		inline static bool is_garbage(const Node<W>* p_node) {
			if (p_node) {
				return p_node->m_ref_count.load(std::memory_order::memory_order_relaxed) == 0;
			}
			else {
				return false;
			}
		}

		static void ref_inc(Node<W>* p_node) {
			if (p_node) {
				(p_node->m_ref_count)++;
				if (p_node->m_ref_count.load(std::memory_order::memory_order_relaxed) == 1) {
					for (auto&& succ : p_node->m_successors) {
						ref_inc(succ.get_node());
					}
				}
			}

		}

		static void ref_dec(Node<W>* p_node) {
			if (p_node) {
				(p_node->m_ref_count)--;
				if (p_node->m_ref_count.load(std::memory_order::memory_order_relaxed) == 0) {
					for (auto&& succ : p_node->m_successors) {
						ref_dec(succ.get_node());
					}
				}
				auto t = p_node->m_ref_count.load(std::memory_order::memory_order_relaxed);
				assert(p_node->m_ref_count.load(std::memory_order::memory_order_relaxed) >= 0);
			}
		}

		/// <summary>
		/// clear the unique_table
		/// id is rearranged.
		/// </summary>
		/// <returns> </returns>
		static void reset() {
			for (auto&& i : *mp_unique_table) {
				delete i.second;
			}

			delete mp_unique_table;
			mp_unique_table = new cache::unique_table<W>{};
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
				std::cout << "|  " << j << " " << "node: " << m_successors[j].get_node() << std::endl;
			}

			for (const auto& succ : m_successors) {
				if (succ.get_node() != nullptr) {
					succ.get_node()->print();
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
			auto&& node_ls = boost::unordered_set<const Node<W>*>{};
			node_search(node_ls);
			// the terminal node is counted
			return node_ls.size() + 1;
		}

		inline static const cache::unique_table<W>* get_unique_table() {
			return mp_unique_table;
		}
	};


	template <typename W>
	struct weightednode {
	public:
		W weight;
	private:
		Node<W>* node;

	public:

		inline Node<W>* get_node() const {
			return node;
		}

		inline void set_node(Node<W>* _node) {
			Node<W>::ref_dec(node);
			node = _node;
			Node<W>::ref_inc(node);
		}

		weightednode(const weightednode<W>& other) {
			weight = other.weight;
			node = other.node;
			Node<W>::ref_inc(node);
		}

		weightednode(weightednode<W>&& other) {
			node = nullptr;
			*this = std::move(other);
		}

		weightednode(W&& _weight, node::Node<W>* _p_node) {
			weight = std::move(_weight);
			node = _p_node;
			Node<W>::ref_inc(node);
		}

		weightednode(): node(nullptr) {}

		weightednode& operator = (const weightednode& other) {
			weight = other.weight;
			Node<W>::ref_dec(node);
			node = other.node;
			Node<W>::ref_inc(node);
			return *this;
		}

		weightednode& operator = (weightednode&& other) {
			weight = std::move(other.weight);
			auto temp = node;
			node = other.node;
			other.node = temp;
			return *this;
		}

		template <bool PL>
		static weightednode<W> get_wnode(W&& wei, int order, succ_ls<W>&& successors) {
			auto&& key = cache::unique_table_key<W>(order, successors);

			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			if constexpr (PL) {
				Node<W>::unique_table_m.lock();
			}
			auto&& p_find_res = Node<W>::mp_unique_table->find(key);

			if (p_find_res != Node<W>::mp_unique_table->end()) {
				// and another reference
				Node<W>::ref_inc(p_find_res->second);
				if constexpr (PL) {
					Node<W>::unique_table_m.unlock();
				}

				//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

				// construct at once to avoid accidental clean of p_find_res->second, (the node may have reference count of 0)
				weightednode<W> res;
				res.weight = std::move(wei);
				res.node = p_find_res->second;
				return res;
			}

			// a node of reference 1 is created.
			node::Node<W>* p_node = new node::Node<W>(order, std::move(successors));

			(*Node<W>::mp_unique_table)[key] = p_node;
			if constexpr (PL) {
				Node<W>::unique_table_m.unlock();
			}
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			weightednode<W> res;
			res.weight = std::move(wei);
			res.node = p_node;
			return res;
		}


		inline bool isterminal() const {
			return node == nullptr;
		}

		inline bool is_multi_ref() const {
			return node->m_ref_count.load(std::memory_order::memory_order_relaxed) > 1;
		}

		~weightednode() {
			Node<W>::ref_dec(node);
		}
	};

	template <typename W>
	struct wnode_cache {
	public:
		W weight;
	private:
		Node<W>* node;

	public:
		inline Node<W>* get_node() const {
			return node;
		}

		wnode_cache(const weightednode<W>& w_node) {
			weight = w_node.weight;
			node = w_node.get_node();
		}

		wnode_cache() : node(nullptr) {}

		wnode_cache& operator = (const weightednode<W>& w_node) {
			weight = w_node.weight;
			node = w_node.get_node();
			return *this;
		}

		inline bool is_garbage() const {
			return node->is_garbage();
		}

		inline weightednode<W> weightednode() const {
			return node::weightednode<W>(W{ weight }, node);
		}
	};
}