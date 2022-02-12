#include "CUDAcpl.h"

using namespace std;
using namespace CUDAcpl;


Tensor CUDAcpl::mul_element_wise(const Tensor& t, Complex s) {
	auto dim = t.dim() - 1;
	auto t_real = t.select(dim, 0);
	auto t_imag = t.select(dim, 1);
	auto res_real = t_real * s.real() - t_imag * s.imag();
	auto res_imag = t_real * s.imag() + t_imag * s.real();
	return torch::stack({ res_real, res_imag }, dim);
}

Tensor CUDAcpl::tensordot(const Tensor& a, const Tensor& b,
	c10::IntArrayRef dim_self, c10::IntArrayRef dim_other) {
	auto a_dim = a.dim() - 1;
	auto a_real = a.select(a_dim, 0);
	auto a_imag = a.select(a_dim, 1);
	auto b_dim = b.dim() - 1;
	auto b_real = b.select(b_dim, 0);
	auto b_imag = b.select(b_dim, 1);
	auto res_real = torch::tensordot(a_real, b_real, dim_self, dim_other) -
		torch::tensordot(a_imag, b_imag, dim_self, dim_other);
	auto res_imag = torch::tensordot(a_real, b_imag, dim_self, dim_other) +
		torch::tensordot(a_imag, b_real, dim_self, dim_other);
	return torch::stack({ res_real, res_imag }, res_real.dim());
}

