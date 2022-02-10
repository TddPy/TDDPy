#include "Node.h"
#include "weights.h"

using namespace dict;
using namespace node;

bool dict::operator==(const unique_table_key& a, const unique_table_key& b) {
	if (a.order == b.order &&
		a.range == b.range) {
		double eps = Node::EPS();
		for (int i = 0; i < a.range; i++) {
			if (weights::is_equal(a.p_weights[i], b.p_weights[i], eps) && a.p_nodes[i] == b.p_nodes[i]) {
				continue;
			}
			else {
				return false;
			}
		}
		return true;
	}
	else {
		return false;
	}
}

std::size_t dict::hash_value(const unique_table_key& key_struct) {
	std::size_t seed = 0;
	boost::hash_combine(seed, key_struct.order);
	for (int i = 0; i < key_struct.range; i++) {
		boost::hash_combine(seed, Node::get_int_key(key_struct.p_weights[i].real()));
		boost::hash_combine(seed, Node::get_int_key(key_struct.p_weights[i].imag()));
	}
	for (int i = 0; i < key_struct.range; i++) {
		boost::hash_combine(seed, key_struct.p_nodes[i]);
	}
	return seed;
}


/*
	Implementations of Node.
*/

double Node::m_EPS = DEFAULT_EPS;
int Node::m_global_id = 0;
unique_table Node::m_unique_table = unique_table();

void Node::node_search(std::vector<node_int>& id_ls) const {
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



double Node::EPS() {
	return m_EPS;
}

void Node::reset() {
	m_unique_table = dict::unique_table();
	m_global_id = 0;
}

int Node::get_int_key(double weight) {
	return (int)round(weight / m_EPS);
}

Node::Node(int id, node_int order, node_int range, wcomplex* p_weights, const Node** p_successors) {
	m_id = id;
	m_order = order;
	m_range = range;
	mp_weights = p_weights;
	mp_successors = p_successors;
}

Node::~Node() {
	free(mp_weights);
	free(mp_successors);
}

node_int Node::get_order() const {
	return m_order;
}

node_int Node::get_range() const {
	return m_range;
}

const wcomplex* Node::get_weights() const {
	return mp_weights;
}

const Node** Node::get_successors() const {
	return mp_successors;
}

size_t Node::get_size() const {
	auto id_ls = std::vector<node_int>();
	node_search(id_ls);
	return id_ls.size();
}

dict::unique_table_key Node::get_key_struct() const {
	dict::unique_table_key key_struct = {
		m_order,
		m_range,
		mp_weights,
		mp_successors
	};
	return key_struct;
}

std::size_t Node::get_hash(Node* p_node) {
	if (p_node == TERMINAL_NODE) {
		return 0;
	}
	std::size_t seed = 0;
	return dict::hash_value(p_node->get_key_struct());
}

Node* Node::get_unique_node(node_int order, node_int range, wcomplex* p_weights, const Node** p_successors) {
	dict::unique_table_key key_struct = {
		order,
		range,
		p_weights,
		p_successors
	};
	auto p_res = m_unique_table.find(key_struct);
	if (p_res != m_unique_table.end()) {
		free(p_weights);
		free(p_successors);
		return p_res->second;
	}

	Node* p_node = new Node(m_global_id, order, range, p_weights, p_successors);
	m_global_id += 1;
	m_unique_table[key_struct] = p_node;
	return p_node;
}

//Node* const TERMINAL_NODE = nullptr;