#include "Node.h"
#include "weights.h"

using namespace dict;
using namespace node;

unique_table_key::unique_table_key(node_int _order, node_int _range, const wcomplex* _p_weights, const node::Node** _p_nodes) {
	order = _order;
	range = _range;
	p_weights_real = (int*)malloc(sizeof(int) * range);
	p_weights_imag = (int*)malloc(sizeof(int) * range);
	for (int i = 0; i < range; i++) {
		p_weights_real[i] = Node::get_int_key(_p_weights[i].real());
		p_weights_imag[i] = Node::get_int_key(_p_weights[i].imag());
	}
	p_nodes = _p_nodes;
}
unique_table_key&  unique_table_key::operator =(const unique_table_key& other) {
	if (range != other.range) {
		range = other.range;
		free(p_weights_real);
		free(p_weights_imag);
		p_weights_real = (int*)malloc(sizeof(int) * range);
		p_weights_imag = (int*)malloc(sizeof(int) * range);
	}
	order = other.order;
	p_nodes = other.p_nodes;
	for (int i = 0; i < range; i++) {
		p_weights_real[i] = other.p_weights_real[i];
		p_weights_imag[i] = other.p_weights_imag[i];
	}
	return *this;
}

unique_table_key::unique_table_key(const unique_table_key& other) {
	order = other.order;
	range = other.range;
	p_weights_real = array_clone(other.p_weights_real, range);
	p_weights_imag = array_clone(other.p_weights_imag, range);
	p_nodes = other.p_nodes;
}

unique_table_key::~unique_table_key() {
#ifdef DECONSTRUCTOR_DEBUG
	if (p_weights_real == nullptr || p_weights_imag == nullptr) {
		std::cout << "unique_table_key repeat deconstruction" << std::endl;
	}
	free(p_weights_real);
	p_weights_real = nullptr;
	free(p_weights_imag);
	p_weights_imag = nullptr;
#elif
	free(p_weights_real);
	free(p_weights_imag);
#endif
}

bool dict::operator==(const unique_table_key& a, const unique_table_key& b) {
	if (a.order == b.order &&
		a.range == b.range) {
		double eps = Node::EPS();
		for (int i = 0; i < a.range; i++) {
			if (a.p_weights_real[i] == b.p_weights_real[i] && 
				a.p_weights_imag[i] == b.p_weights_imag[i] &&
				a.p_nodes[i] == b.p_nodes[i]) {
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
		boost::hash_combine(seed, key_struct.p_weights_real[i]);
		boost::hash_combine(seed, key_struct.p_weights_imag[i]);
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

const Node* Node::duplicate_iterate(const Node* p_node, int order_shift, dict::duplicate_table& duplicate_cache) {
	if (p_node == TERMINAL_NODE) {
		return TERMINAL_NODE;
	}

	auto order = p_node->m_order + order_shift;
	auto key = Node::get_id_all(p_node);
	auto p_find_res = duplicate_cache.find(key);
	if (p_find_res != duplicate_cache.end()) {
		return p_find_res->second;
	}
	else {
		//broadcast to contain the extra shape
		//...

		
		wcomplex* p_weights = array_clone<wcomplex>(p_node->mp_weights, p_node->m_range);
		const Node** p_successors = (const Node**)malloc(sizeof(const Node*) * p_node->m_range);
		for (int i = 0; i < p_node->m_range; i++) {
			p_successors[i] = Node::duplicate_iterate(p_node->mp_successors[i], order_shift, duplicate_cache);
		}
		
		auto p_res = Node::get_unique_node(order, p_node->m_range, p_weights, p_successors);
		duplicate_cache.insert(std::make_pair(key, p_res));
		return p_res;
	}
}

const Node* Node::append_iterate(const Node* p_a, const Node* p_b) {
	if (p_a == TERMINAL_NODE) {
		return p_b;
	}

	wcomplex* p_weights = array_clone<wcomplex>(p_a->mp_weights, p_a->m_range);
	const Node** p_successors = (const Node**)malloc(sizeof(const Node*) * p_a->m_range);
	for (int i = 0; i < p_a->m_range; i++) {
		p_successors[i] = Node::append_iterate(p_a->mp_successors[i], p_b);
	}
	return Node::get_unique_node(p_a->m_order, p_a->m_range, p_weights, p_successors);
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
#ifdef DECONSTRUCTOR_DEBUG
	if (mp_weights == nullptr || mp_successors == nullptr) {
		std::cout << "Node repeat deconstruction" << std::endl;
	}
	free(mp_weights);
	mp_weights = nullptr;
	free(mp_successors);
	mp_successors = nullptr;
#elif
	free(mp_weights);
	free(mp_successors);
#endif
}

int Node::get_id() const {
	return m_id;
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
	auto* p_key_struct = new dict::unique_table_key(m_order, m_range, mp_weights, mp_successors);
	return *p_key_struct;
}

std::size_t Node::get_hash(const Node* p_node) {
	if (p_node == TERMINAL_NODE) {
		return 0;
	}
	std::size_t seed = 0;
	return dict::hash_value(p_node->get_key_struct());
}

int Node::get_id_all(const Node* p_node) {
	if (p_node == TERMINAL_NODE) {
		return TERMINAL_NODE_ID;
	}
	else {
		return p_node->m_id;
	}
}

const Node* Node::get_unique_node(node_int order, node_int range, wcomplex* p_weights, const Node** p_successors) {
	auto key_struct = dict::unique_table_key(order, range, p_weights, p_successors);
	auto p_res = m_unique_table.find(key_struct);
	if (p_res != m_unique_table.end()) {
		free(p_weights);
		free(p_successors);
		return p_res->second;
	}

	Node* p_node = new Node(m_global_id, order, range, p_weights, p_successors);
	m_global_id += 1;
	m_unique_table.insert(std::make_pair(key_struct, p_node));
	return p_node;
}

const Node* Node::duplicate(const Node* p_node, int order_shift) {
	auto duplicate_cache = dict::duplicate_table();
	return Node::duplicate_iterate(p_node, order_shift, duplicate_cache);
}



const Node* Node::append(const Node* p_a, int a_depth, const Node* p_b, bool parallel_tensor) {
	if (!parallel_tensor) {
		auto modified_node = Node::duplicate(p_b, a_depth);
		return Node::append_iterate(p_a, modified_node);
	}
	else {
		//not implemented yet
		throw - 1;
	}
}
