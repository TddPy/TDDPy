#include "CUDAcpl.h"

using namespace std;
using namespace CUDAcpl;

CUDAcpl::Tensor CUDAcpl::from_complex(Complex cpl) {
	auto res = torch::empty({ 2 });
	res[0] = cpl.real();
	res[1] = cpl.imag();
	return res;
}


Complex CUDAcpl::item(const Tensor& t) {
	return Complex(t.index({ "...",0 }).cpu().item().toDouble(),
		t.index({ "...",1 }).cpu().item().toDouble());
}

CUDAcpl::Tensor CUDAcpl::ones(c10::IntArrayRef size) {
	auto real = torch::ones(size);
	auto imag = torch::zeros(size);
	return torch::stack({ real, imag }, size.size());
}

CUDAcpl::Tensor CUDAcpl::mul_element_wise(const Tensor& t, Complex s) {
	auto dim = t.dim();
	auto t_real = t.select(dim - 1, 0);
	auto t_imag = t.select(dim - 1, 1);
	auto res_real = t_real * s.real() - t_imag * s.imag();
	auto res_imag = t_real * s.imag() + t_imag * s.real();
	return torch::stack({ res_real, res_imag }, dim-1);
}
