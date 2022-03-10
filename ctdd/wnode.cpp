#include "stdafx.h"
#include "wnode.hpp"

wcomplex wnode<wcomplex>::get_normalizer(const node::succ_ls<wcomplex>& successors) {
	int i_max = 0;
	double norm_max = norm(successors[0].weight);
	for (int i = 1; i < successors.size(); i++) {
		double temp_norm = norm(successors[i].weight);
		// alter the maximum according to EPS, to avoid arbitrary normalization
		if (temp_norm - norm_max > weight::EPS) {
			i_max = i;
			norm_max = temp_norm;
		}
	}
	return successors[i_max].weight;
}

CUDAcpl::Tensor wnode<CUDAcpl::Tensor>::get_normalizer(const node::succ_ls<CUDAcpl::Tensor>& successors) {
	auto&& sizes = successors[0].weight.sizes();
	std::vector<int64_t> temp_shape(sizes.begin(), sizes.end());
	temp_shape.erase(temp_shape.end() - 1);
	auto norm_max = CUDAcpl::norm(successors[0].weight);

	auto normalizer = successors[0].weight;

	for (int i = 1; i < successors.size(); i++) {
		auto temp_norm = CUDAcpl::norm(successors[i].weight);
		auto alter_matrix = temp_norm - norm_max > weight::EPS;
		norm_max = torch::where(alter_matrix, temp_norm, norm_max);

		auto all_alter_matrix = alter_matrix.unsqueeze(alter_matrix.dim());
		all_alter_matrix = all_alter_matrix.expand_as(normalizer);
		normalizer = torch::where(all_alter_matrix, successors[i].weight, normalizer);
	}
	
	// process the 0-elements

	auto zero_item = (norm_max < weight::EPS).unsqueeze(normalizer.dim() - 1);
	zero_item = zero_item.expand_as(normalizer);
	normalizer = torch::where(zero_item, CUDAcpl::ones_like(normalizer), normalizer);
	return normalizer;
}
