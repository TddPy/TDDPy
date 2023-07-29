/*
    All the type information and synonyms are defined here.

*/

#pragma once

#include <xtensor/xarray.hpp>

namespace node {
	template <typename T_WEIGHT>
	class Node;
}


namespace wnode {
    template <typename T_WEIGHT>
    struct WNode;
}


// succ_ls : type synonym for list of successors (weighted nodes)
template <typename T_WEIGHT>
using succ_ls = std::vector<wnode::WNode<T_WEIGHT>>;

// We stick to the single case for now.
typedef xt::xarray<std::complex<double>> Tensor;
