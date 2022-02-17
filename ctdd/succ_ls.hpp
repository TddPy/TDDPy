#pragma once
#include "stdafx.h"

namespace node {
	template <class W>
	class Node;

	template <class W>
	struct weightednode {
		W weight;
		const Node<W>* node;

		weightednode(const weightednode<W>& other) {
			weight = other.weight;
			node = other.node;
		}

		weightednode(weightednode<W>&& other) {
			weight = std::move(other.weight);
			node = other.node;
		}

		weightednode(const W& _weight, const node::Node<W>* _p_node) {
			weight = _weight;
			node = _p_node;
		}

		weightednode(W&& _weight, const node::Node<W>* _p_node) {
			weight = std::move(_weight);
			node = _p_node;
		}

		weightednode() {}

		weightednode& operator = (const weightednode& other) {
			weight = other.weight;
			node = other.node;
			return *this;
		}

		weightednode& operator = (weightednode&& other) {
			weight = std::move(other.weight);
			node = other.node;
			return *this;
		}

		inline bool isterminal() const {
			return node == nullptr;
		}
	};

	template <class W>
	using succ_ls = std::vector<weightednode<W>>;
}