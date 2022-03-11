#include "stdafx.h"
#include "wnode.hpp"
using namespace std;

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
	auto norm_max = CUDAcpl::norm(successors[0].weight);

	auto&& normalizer = successors[0].weight.clone();

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

weight::sum_nweights<wcomplex> wnode<wcomplex>::weights_normalize(const wcomplex& weight1, const wcomplex& weight2) {
	wcomplex renorm_coef = (norm(weight2) - norm(weight1) > weight::EPS) ? weight2 : weight1;

	auto&& nweight1 = wcomplex(0., 0.);
	auto&& nweight2 = wcomplex(0., 0.);
	if (norm(renorm_coef) > weight::EPS) {
		nweight1 = weight1 / renorm_coef;
		nweight2 = weight2 / renorm_coef;
	}
	else {
		renorm_coef = wcomplex(1., 0.);
	}
	return weight::sum_nweights<wcomplex>(
		std::move(nweight1),
		std::move(nweight2),
		std::move(renorm_coef)
		);
}

weight::sum_nweights<CUDAcpl::Tensor> wnode<CUDAcpl::Tensor>::weights_normalize(
	const CUDAcpl::Tensor& weight1, const CUDAcpl::Tensor& weight2) {
	auto norm2 = CUDAcpl::norm(weight2);
	auto norm1 = CUDAcpl::norm(weight1);
	auto chose_2 = norm2 - norm1 > weight::EPS;
	auto norm_max = torch::where(chose_2, norm2, norm1);

	auto all_chose_2 = chose_2.unsqueeze(chose_2.dim());
	all_chose_2 = all_chose_2.expand_as(weight1);

	auto renorm_coef = where(all_chose_2, weight2, weight1);

	// process the 0-elements
	auto zero_item = (norm_max < weight::EPS).unsqueeze(renorm_coef.dim() - 1);
	zero_item = zero_item.expand_as(renorm_coef);

	renorm_coef = where(zero_item, CUDAcpl::ones_like(renorm_coef), renorm_coef);
	auto reciprocal = CUDAcpl::reciprocal(renorm_coef);
	
	auto nweight1 = CUDAcpl::mul_element_wise(weight1, reciprocal);
	auto nweight2 = CUDAcpl::mul_element_wise(weight2, reciprocal);

	return weight::sum_nweights<CUDAcpl::Tensor>(
		std::move(nweight1),
		std::move(nweight2),
		std::move(renorm_coef)
		);
}
