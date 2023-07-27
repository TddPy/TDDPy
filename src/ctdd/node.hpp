#pragma once

#include <vector>
#include "cache.hpp"

namespace node {

	template <class W>
	using succ_ls = std::vector<WNode<W>>;


	// The node used in tdd.
	template <typename W>
	class Node {
	private:

		// The unique_table to store all the node instances used in tdd.
		static cache::unique_table<W> m_unique_table;

		// represent the order of this node (which tensor index it represent)
		const int m_order;

		/* The weight and node of the successors
		*  Note: terminal nodes are represented by nullptr in the successors.
		*/
		succ_ls<W> m_successors;


	public:

		template <typename T>
		friend class WNode;

		inline int get_order() const noexcept {
			return m_order;
		}

		inline int get_range() const noexcept {
			return m_successors.size();
		}

		inline const succ_ls<W>& get_successors() const noexcept {
			return m_successors;
		}

		inline static const cache::unique_table<W> get_unique_table() {
			return m_unique_table;
		}
	};


	template <typename W>
	struct WNode {
	public:
		W weight;
	private:
		Node<W>* m_p_node;

	public:

		inline Node<W>* get_node() const noexcept {
			return m_p_node;
		}

		inline void set_node(Node<W>* p_node) noexcept {
			m_p_node = p_node;
		}

		WNode(const WNode<W>& other) noexcept {
			weight = other.weight;
			node = other.node;
			Node<W>::ref_inc(node);
		}

		WNode(WNode<W>&& other) noexcept {
			node = nullptr;
			*this = std::move(other);
		}

		WNode(W&& _weight, node::Node<W>* _p_node) noexcept {
			weight = std::move(_weight);
			node = _p_node;
		}

		WNode() noexcept : node(nullptr) {}

		WNode& operator = (const WNode& other) noexcept {
			weight = other.weight;
			node = other.node;
			return *this;
		}

		WNode& operator = (WNode&& other) noexcept {
			weight = std::move(other.weight);
			auto temp = node;
			node = other.node;
			other.node = temp;
			return *this;
		}

		static WNode<W> get_wnode(W&& wei, int order, succ_ls<W>&& successors) {
			auto&& key = cache::unique_table_key<W>(order, successors);

			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			Node<W>::unique_table_m.lock();
			auto&& p_find_res = Node<W>::m_unique_table.find(key);

			if (p_find_res != Node<W>::m_unique_table.end()) {
				// and another reference
				Node<W>::ref_inc(p_find_res->second);
				Node<W>::unique_table_m.unlock();
				//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

				// construct at once to avoid accidental clean of p_find_res->second, (the node may have reference count of 0)
				weightednode<W> res;
				res.weight = std::move(wei);
				res.node = p_find_res->second;
				return res;
			}

			// a node of reference 1 is created.
			node::Node<W>* p_node = new node::Node<W>(order, std::move(successors));

			Node<W>::m_unique_table[key] = p_node;
			Node<W>::unique_table_m.unlock();
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			weightednode<W> res;
			res.weight = std::move(wei);
			res.node = p_node;
			return res;
		}


		inline bool isterminal() const noexcept {
			return node == nullptr;
		}

		inline bool is_multi_ref() const noexcept {
			return Node<W>::is_multi_ref(node);
		}

		~weightednode() noexcept {
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
		inline Node<W>* get_node() const noexcept {
			return node;
		}

		wnode_cache(const weightednode<W>& w_node) noexcept {
			weight = w_node.weight;
			node = w_node.get_node();
		}

		wnode_cache() noexcept : node(nullptr) {}

		wnode_cache& operator = (const weightednode<W>& w_node) noexcept {
			weight = w_node.weight;
			node = w_node.get_node();
			return *this;
		}

		inline bool is_garbage() const noexcept {
			return Node<W>::is_garbage(node);
		}

		inline weightednode<W> get_weightednode() const noexcept {
			return node::weightednode<W>(W{ weight }, node);
		}
	};
}