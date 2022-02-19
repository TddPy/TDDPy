/*
* It is a small package to implement the use of complex number in torch, by an extra inner dimension.
*/

#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/python.h>

typedef std::complex<double> Complex;

namespace CUDAcpl {
	// The CUDA complex tensor.
	typedef torch::Tensor Tensor;

	inline Tensor from_complex(Complex cpl) {
		auto&& res = torch::empty({ 2 });
		res[0] = cpl.real();
		res[1] = cpl.imag();
		return res;
	}

	inline Complex item(const Tensor& t) {
		return Complex(t.index({ "...",0 }).cpu().item().toDouble(),
			t.index({ "...",1 }).cpu().item().toDouble());
	}

	inline Tensor ones(c10::IntArrayRef size) {
		auto&& real = torch::ones(size);
		auto&& imag = torch::zeros(size);
		return torch::stack({ real, imag }, size.size());
	}

	Tensor tensordot(const Tensor& a, const Tensor& b,
		c10::IntArrayRef dim_self, c10::IntArrayRef dim_other);

	CUDAcpl::Tensor mul_element_wise(const Tensor& t, Complex s);
}