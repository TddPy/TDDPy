#include "CUDAcpl.h"

using namespace std;
using namespace CUDAcpl;


CUDAcpl::Tensor CUDAcpl::reciprocal_without_zero(const Tensor& a) {
	auto&& a_dim = a.dim() - 1;
	auto&& a_real = a.select(a_dim, 0);
	auto&& a_imag = a.select(a_dim, 1);
	auto&& res = torch::stack({ a_real, -a_imag }, a_dim);
	auto&& denominator = (a_real * a_real + a_imag * a_imag);
	denominator = torch::where(denominator.toType(c10::ScalarType::Bool),
		denominator, torch::ones_like(denominator, tensor_opt));
	denominator = denominator.unsqueeze(a_dim).expand_as(res);
	return res / denominator;
}

Tensor CUDAcpl::tensordot(const Tensor& a, const Tensor& b,
	c10::IntArrayRef dim_self, c10::IntArrayRef dim_other) {
	auto&& a_dim = a.dim() - 1;
	auto&& a_real = a.select(a_dim, 0);
	auto&& a_imag = a.select(a_dim, 1);
	auto&& b_dim = b.dim() - 1;
	auto&& b_real = b.select(b_dim, 0);
	auto&& b_imag = b.select(b_dim, 1);
	auto&& res_real = torch::tensordot(a_real, b_real, dim_self, dim_other) -
		torch::tensordot(a_imag, b_imag, dim_self, dim_other);
	auto&& res_imag = torch::tensordot(a_real, b_imag, dim_self, dim_other) +
		torch::tensordot(a_imag, b_real, dim_self, dim_other);
	return torch::stack({ res_real, res_imag }, res_real.dim());
}

Tensor CUDAcpl::einsum(c10::string_view equation, at::TensorList tensors) {
	auto&& a = tensors[0];
	auto&& b = tensors[1];
	auto&& a_dim = a.dim() - 1;
	auto&& a_real = a.select(a_dim, 0);
	auto&& a_imag = a.select(a_dim, 1);
	auto&& b_dim = b.dim() - 1;
	auto&& b_real = b.select(b_dim, 0);
	auto&& b_imag = b.select(b_dim, 1);
	auto&& res_real = torch::einsum(equation, { a_real, b_real }) - torch::einsum(equation, { a_imag, b_imag });
	auto&& res_imag = torch::einsum(equation, { a_real, b_imag }) + torch::einsum(equation, { a_imag, b_real });
	return torch::stack({ res_real, res_imag }, res_real.dim());
}
