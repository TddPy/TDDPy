#pragma once
#include "stdafx.h"
#include "cache.h"




namespace node {

	//The node used in tdd.
	template <class W>
	class Node
	{
	private:

		// record the size of m_unique_table
		static int m_global_id;

		// The unique_table to store all the node instances used in tdd.
		static cache::unique_table<W> m_unique_table;

		int m_id;

		//represent the order of this node (which tensor index it represent)
		node_int m_order;

		// the number of possible values of this index
		node_int m_range;

		W* mp_weights;

		// Note: terminal nodes are represented by nullptr in the successors.
		const Node<W>** mp_successors;


	private:
		/// <summary>
		/// Count all the nodes starting from this node.
		/// </summary>
		/// <param name="total"> current total ids in p_id </param>
		/// <param name="p_id"> the memory to store all the ids (it is a borrowed pointer) </param>
		void node_search(std::vector<node_int>& id_ls) const;

		static const Node<W>* duplicate_iterate(const node::Node<W>* p_node, int order_shift,
			int dim_para_shape, const int64_t* p_parallel_shape,
			int dim_shape_front, const int64_t* p_shape_front,
			int dim_shape_back, const int64_t* p_shape_back, cache::duplicate_table<W>& duplicate_cache);


		static const Node<W>* shift_multiple_iterate(const Node<W>* p_node, int length, const int* p_new_order, cache::duplicate_table<W>& shift_cache);


		/// <summary>
		/// Direct append without any further operation. Only used within other methods (like Node::append).
		/// </summary>
		/// <param name="p_a"></param>
		/// <param name="p_b"></param>
		/// <returns></returns>
		static const Node<W>* append_iterate(const Node<W>* p_a, const Node<W>* p_b);
	public:
		// Reset the dictionary caches.
		static void reset();

		/// <summary>
		/// Constructor for new nodes (inner use only, otherwise refer to unique_table)
		/// Note that the dynanmically allocated p_out_weights and p_successors will be owned by this node.
		/// </summary>
		/// <param name="id"></param>
		/// <param name="order"></param>
		/// <param name="range"></param>
		/// <param name="p_weights">[ownership transfer]</param>
		/// <param name="p_successors">[ownership transfer]</param>
		Node(int id, node_int order, node_int range, W* p_weights, const Node<W>** p_successors);

		~Node();

		int get_id() const;
		node_int get_order() const;
		node_int get_range() const;
		const W* get_weights() const;
		const Node<W>** get_successors() const;

		/// <summary>
		/// Count all the nodes starting from this one.
		/// </summary>
		/// <returns></returns>
		size_t get_size() const;

		/// <summary>
		/// Get the corresponding key structure of this node.
		/// </summary>
		/// <returns></returns>
		cache::unique_table_key<W> get_key_struct() const;

		/// <summary>
		/// Calculate and return the Hash value of the node (can be nullptr).
		/// </summary>
		/// <param name="p_node"></param>
		/// <returns></returns>
		static std::size_t get_hash(const Node<W>* p_node);

		/// <summary>
		/// Get ID for all nodes (including terminal nodes)
		/// </summary>
		/// <param name="p_node"></param>
		/// <returns></returns>
		static int get_id_all(const Node<W>* p_node);

		/// <summary>
		/// Return the required node. It is either from the unique table, or a newly created one.
		/// Note : The equality checking inside is conducted with the node.EPS tolerance.So feel free
		/// to pass in the raw weights from calculation.
		/// </summary>
		/// <param name="order">represent the order of this node(which tensor index it represent)</param>
		/// <param name="range">the count of possible value</param>
		/// <param name="p_weights">[onwership transfer] the weights of this node</param>
		/// <param name="p_successors">[onwership transfer] the successor nodes</param>
		/// <returns></returns>
		static const Node<W>* get_unique_node(node_int order, node_int range, W* p_weights, const Node<W>** p_successors);

		/// <summary>
		/// Duplicate from this node, with the initial order of (node.order + order_shift),
		///	and broadcast it to contain the extra(parallel index) shape aheadand behind.
		/// </summary>
		/// <param name="p_node"></param>
		/// <param name="order_shift"></param>
		/// <returns></returns>
		static const Node<W>* duplicate(const node::Node<W>* p_node, int order_shift,
			int dim_para_shape, const int64_t* p_parallel_shape,
			int dim_shape_front, const int64_t* p_shape_front,
			int dim_shape_back, const int64_t* p_shape_back);

		/// <summary>
		/// Shift the order of node, Return the result.
		/// order of new node is p_new_order[node.order]
		/// </summary>
		/// <param name="p_node"></param>
		/// <param name="length"></param>
		/// <param name="p_new_order">[borrowed]</param>
		/// <returns></returns>
		static const Node<W>* shift_multiple(const Node<W>* p_node, int length, const int* p_new_order);


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
		static const Node<W>* append(const Node<W>* p_a, int a_depth, const Node<W>* p_b, bool parallel_tensor = false);
	};

	// The pointer of the terminal node (nullptr).
	void* const TERMINAL_NODE = nullptr;
	const int TERMINAL_NODE_ID = -1;







	/* Implementation below.
	*/

	template <class W>
	void node::Node<W>::node_search(std::vector<node_int>& id_ls) const {
		// check whether it is in p_id already
		for (auto i = id_ls.begin(); i != id_ls.end(); i++) {
			if (*i == m_id) {
				return;
			}
		}
		// it is not counted yet in this case
		id_ls.push_back(m_id);
		for (int i = 0; i < m_range; i++) {
			if (mp_successors[i] != nullptr) {
				mp_successors[i]->node_search(id_ls);
			}
		}
	}

	template <class W>
	const node::Node<W>* node::Node<W>::duplicate_iterate(const node::Node<W>* p_node, int order_shift, 
		int dim_para_shape, const int64_t* p_parallel_shape,
		int dim_shape_front, const int64_t* p_shape_front,
		int dim_shape_back, const int64_t* p_shape_back, cache::duplicate_table<W>& duplicate_cache) {
		if (p_node == (const node::Node<W>*)TERMINAL_NODE) {
			return (const node::Node<W>*)TERMINAL_NODE;
		}

		auto order = p_node->m_order + order_shift;
		auto key = node::Node<W>::get_id_all(p_node);
		auto p_find_res = duplicate_cache.find(key);
		if (p_find_res != duplicate_cache.end()) {
			return p_find_res->second;
		}
		else {
			W* p_weights = array_clone<W>(p_node->mp_weights, p_node->m_range);

			//for CUDAcpl Tensor representatoin, broadcast to contain the extra shape
			if (std::is_same<W, CUDAcpl::Tensor>::value) {
				auto len = dim_para_shape + dim_shape_front + dim_shape_back;
				auto p_temp_shape = (int64_t*)malloc(sizeof(int64_t) * (len + 1));
				p_temp_shape[len] = 2;
				//unsqueeze
				for (int i = 0; i < dim_shape_front; i++) {
					p_temp_shape[i] = 1;
				}
				for (int i = 0; i < dim_para_shape; i++) {
					p_temp_shape[i + dim_shape_front] = p_parallel_shape[i];
				}
				for (int i = 0; i < dim_shape_back; i++) {
					p_temp_shape[i + dim_shape_front + dim_para_shape] = 1;
				}
				c10::IntArrayRef temp_shape = c10::IntArrayRef(p_temp_shape, len + 1);
				for (int i = 0; i < m_range; i++) {
					p_weights[i] = p_weights[i].view(temp_shape);
				}
				//expand
				for (int i = 0; i < dim_shape_front; i++) {
					p_temp_shape[i] = p_shape_front[i];
				}
				for (int i = 0; i < dim_shape_back; i++) {
					p_temp_shape[i + dim_shape_front + dim_para_shape] = p_shape_back[i];
				}
				temp_shape = c10::IntArrayRef(p_temp_shape, len + 1);
				for (int i = 0; i < m_range; i++) {
					p_weights[i] = p_weights[i].expand(temp_shape);
				}
				free(p_temp_shape);
			}

			const node::Node<W>** p_successors = (const node::Node<W>**)malloc(sizeof(const node::Node<W>*) * p_node->m_range);
			for (int i = 0; i < p_node->m_range; i++) {
				p_successors[i] = node::Node<W>::duplicate_iterate(p_node->mp_successors[i], order_shift, duplicate_cache);
			}

			auto p_res = node::Node<W>::get_unique_node(order, p_node->m_range, p_weights, p_successors);
			duplicate_cache.insert(std::make_pair(key, p_res));
			return p_res;
		}
	}

	template <class W>
	const node::Node<W>* node::Node<W>::shift_multiple_iterate(const node::Node<W>* p_node, int length, const int* p_new_order, cache::duplicate_table<W>& shift_cache) {
		if (p_node == (const node::Node<W>*)TERMINAL_NODE) {
			return (const node::Node<W>*)TERMINAL_NODE;
		}
		auto order = p_new_order[p_node->m_order];
		auto key = node::Node<W>::get_id_all(p_node);
		auto p_find_res = shift_cache.find(key);
		if (p_find_res != shift_cache.end()) {
			return p_find_res->second;
		}
		else {
			auto range = p_node->get_range();
			auto p_successors = p_node->get_successors();
			auto p_weights = array_clone(p_node->get_weights(), range);
			const node::Node<W>** p_nodes = (const node::Node<W>**)malloc(sizeof(const node::Node<W>*) * range);
			for (int i = 0; i < range; i++) {
				p_nodes[i] = shift_multiple_iterate(p_successors[i], length, p_new_order, shift_cache);
			}
			auto p_res_node = node::Node<W>::get_unique_node(order, range, p_weights, p_nodes);
			shift_cache.insert(std::make_pair(key, p_res_node));
			return p_res_node;
		}
	}

	template <class W>
	const node::Node<W>* node::Node<W>::append_iterate(const Node<W>* p_a, const Node<W>* p_b) {
		if (p_a == TERMINAL_NODE) {
			return p_b;
		}

		W* p_weights = array_clone<W>(p_a->mp_weights, p_a->m_range);
		const node::Node<W>** p_successors = (const node::Node<W>**)malloc(sizeof(const node::Node<W>*) * p_a->m_range);
		for (int i = 0; i < p_a->m_range; i++) {
			p_successors[i] = node::Node<W>::append_iterate(p_a->mp_successors[i], p_b);
		}
		return node::Node<W>::get_unique_node(p_a->m_order, p_a->m_range, p_weights, p_successors);
	}

	template <class W>
	void Node<W>::reset() {
		m_unique_table = cache::unique_table<W>();
		m_global_id = 0;
	}

	template <class W>
	Node<W>::Node(int id, node_int order, node_int range, W* p_weights, const Node<W>** p_successors) {
		m_id = id;
		m_order = order;
		m_range = range;
		mp_weights = p_weights;
		mp_successors = p_successors;
	}

	template <class W>
	Node<W>::~Node() {
#ifdef DECONSTRUCTOR_DEBUG
		if (mp_weights == nullptr || mp_successors == nullptr) {
			std::std::cout << "Node repeat deconstruction" << std::std::endl;
		}
		free(mp_weights);
		mp_weights = nullptr;
		free(mp_successors);
		mp_successors = nullptr;
#else
		free(mp_weights);
		free(mp_successors);
#endif
	}

	template <class W>
	int node::Node<W>::get_id() const {
		return m_id;
	}

	template <class W>
	node_int node::Node<W>::get_order() const {
		return m_order;
	}

	template <class W>
	node_int node::Node<W>::get_range() const {
		return m_range;
	}

	template <class W>
	const W* node::Node<W>::get_weights() const {
		return mp_weights;
	}

	template <class W>
	const node::Node<W>** node::Node<W>::get_successors() const {
		return mp_successors;
	}

	template <class W>
	size_t node::Node<W>::get_size() const {
		auto id_ls = std::vector<node_int>();
		node_search(id_ls);
		return id_ls.size();
	}

	template <class W>
	cache::unique_table_key<W> node::Node<W>::get_key_struct() const {
		auto* p_key_struct = new cache::unique_table_key(m_order, m_range, mp_weights, mp_successors);
		return *p_key_struct;
	}

	template <class W>
	std::size_t Node<W>::get_hash(const Node<W>* p_node) {
		if (p_node == TERMINAL_NODE) {
			return 0;
		}
		std::size_t seed = 0;
		return cache::hash_value(p_node->get_key_struct());
	}

	template <class W>
	int Node<W>::get_id_all(const Node<W>* p_node) {
		if (p_node == TERMINAL_NODE) {
			return TERMINAL_NODE_ID;
		}
		else {
			return p_node->m_id;
		}
	}

	template <class W>
	const Node<W>* Node<W>::get_unique_node(node_int order, node_int range, W* p_weights, const Node<W>** p_successors) {
		auto key_struct = cache::unique_table_key<W>(order, range, p_weights, p_successors);
		auto p_res = m_unique_table.find(key_struct);
		if (p_res != m_unique_table.end()) {
			free(p_weights);
			free(p_successors);
			return p_res->second;
		}

		node::Node<W>* p_node = new node::Node<W>(m_global_id, order, range, p_weights, p_successors);
		m_global_id += 1;
		m_unique_table.insert(std::make_pair(key_struct, p_node));
		return p_node;
	}

	template <class W>
	const Node<W>* node::Node<W>::duplicate(const node::Node<W>* p_node, int order_shift,
		int dim_para_shape, const int64_t* p_parallel_shape,
		int dim_shape_front, const int64_t* p_shape_front,
		int dim_shape_back, const int64_t* p_shape_back) {
		return node::Node<W>::duplicate_iterate(p_node, order_shift,
			dim_para_shape, p_parallel_shape,
			dim_shape_front, p_shape_front,
			dim_shape_back, p_shape_back, cache::Global_Cache<W>::duplicate_cache);
	}

	template <class W>
	const Node<W>* node::Node<W>::shift_multiple(const Node<W>* p_node, int length, const int* p_new_order) {
		return node::Node<W>::shift_multiple_iterate(p_node, length, p_new_order, cache::Global_Cache<W>::shift_cache);
	}



	template <class W>
	const Node<W>* Node<W>::append(const Node<W>* p_a, int a_depth, const Node<W>* p_b, bool parallel_tensor) {
		if (!parallel_tensor) {
			auto modified_node = node::Node<W>::duplicate(p_b, a_depth);
			return node::Node<W>::append_iterate(p_a, modified_node);
		}
		else {
			//not implemented yet
			throw - 1;
		}
	}

}

