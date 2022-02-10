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

	CUDAcpl::Tensor from_complex(Complex cpl);

	Complex item(const Tensor& t);

	CUDAcpl::Tensor ones(c10::IntArrayRef size);

	CUDAcpl::Tensor mul_element_wise(const Tensor& t, Complex s);
}