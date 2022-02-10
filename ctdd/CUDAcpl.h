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


	Complex item(const Tensor& t);
}