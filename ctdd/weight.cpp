#include "weight.hpp"
using namespace weight;
using namespace std;

void weight::func<wcomplex>::as_weight(const CUDAcpl::Tensor& t, wcomplex& weight, const std::vector<int64_t>& data_shape) {
	weight = CUDAcpl::item(t);
}

void weight::func<CUDAcpl::Tensor>::as_weight(const CUDAcpl::Tensor& t, CUDAcpl::Tensor& weight, const std::vector<int64_t>& data_shape) {
	std::vector<int64_t> temp_shape{ data_shape };
	temp_shape.push_back(2);
	weight = t.view(temp_shape).clone();
}

CUDAcpl::Tensor weight::func<wcomplex>::from_weight(const wcomplex& weight) {
	return CUDAcpl::from_complex(weight);
}

CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::from_weight(const CUDAcpl::Tensor& weight) {
	return weight;
}

CUDAcpl::Tensor weight::func<wcomplex>::res_mul_weight(const CUDAcpl::Tensor& tensor, const wcomplex& weight) {
	return CUDAcpl::mul_element_wise(tensor, weight);
}

CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::res_mul_weight(
	const CUDAcpl::Tensor& tensor, const CUDAcpl::Tensor& weight) {
	auto&& sizes = tensor.sizes();
	std::vector<int64_t> temp_shape(sizes.begin(), sizes.end());
	for (int i = weight.dim() - 1; i < temp_shape.size() - 1; i++) {
		temp_shape[i] = 1;
	}
	auto weight_expanded = weight.view(temp_shape);
	return CUDAcpl::mul_element_wise(tensor, weight_expanded.expand_as(tensor));
}

wcomplex weight::func<wcomplex>::ones(const std::vector<int64_t>& data_shape) {
	return wcomplex(1., 0.);
}

CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::ones(const std::vector<int64_t>& data_shape) {
	return CUDAcpl::ones(data_shape);
}

wcomplex weight::func<wcomplex>::zeros(const std::vector<int64_t>& data_shape) {
	return wcomplex(0., 0.);
}

CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::zeros(const std::vector<int64_t>& data_shape) {
	return CUDAcpl::zeros(data_shape);
}

wcomplex weight::func<wcomplex>::zeros_like(const wcomplex& weight) {
	return wcomplex(0., 0.);
}
CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::zeros_like(const CUDAcpl::Tensor& weight) {
	return CUDAcpl::zeros_like(weight);
}

bool weight::func<wcomplex>::is_equal(const wcomplex& a, const wcomplex& b) {
	return abs(a.real() - b.real()) < EPS && abs(a.imag() - b.imag()) < EPS;
}

bool weight::func<CUDAcpl::Tensor>::is_equal(const CUDAcpl::Tensor& a, const CUDAcpl::Tensor& b) {
	return  torch::all(torch::abs(a - b) < EPS).item().toBool();
}

bool weight::func<wcomplex>::is_zero(const wcomplex& a) {
	return abs(a.real()) < EPS && abs(a.imag()) < EPS;
}

bool weight::func<CUDAcpl::Tensor>::is_zero(const CUDAcpl::Tensor& a) {
	return torch::all(torch::abs(a)).item().toBool();
}

wcomplex weight::func<wcomplex>::mul(const wcomplex& a, const wcomplex& b) {
	return a * b;
}

CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::mul(const CUDAcpl::Tensor& a, const CUDAcpl::Tensor& b) {
	return CUDAcpl::mul_element_wise(a, b);
}

wcomplex weight::func<wcomplex>::reciprocal(const wcomplex& a) {
	return wcomplex(1., 0.) / a;
}

CUDAcpl::Tensor weight::func<CUDAcpl::Tensor>::reciprocal(const CUDAcpl::Tensor& a) {
	return CUDAcpl::reciprocal(a);
}


