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

	extern c10::TensorOptions tensor_opt;

	inline void reset(bool device_cuda) {
		if (device_cuda) {
			tensor_opt = tensor_opt.device(c10::Device::Type::CUDA);
		}
		else {
			tensor_opt = tensor_opt.device(c10::Device::Type::CPU);
		}

		tensor_opt = tensor_opt.dtype(c10::ScalarType::Double);
	}

	inline Tensor from_complex(Complex cpl) {
		auto&& res = torch::empty({ 2 }, tensor_opt);
		res[0] = cpl.real();
		res[1] = cpl.imag();
		return res;
	}

	inline Complex item(const Tensor& t) {
		return Complex(t.index({ "...",0 }).cpu().item().toDouble(),
			t.index({ "...",1 }).cpu().item().toDouble());
	}

	inline Tensor norm(const Tensor& t) {
		auto&& t_dim = t.dim() - 1;
		auto&& t_real = t.select(t_dim, 0);
		auto&& t_imag = t.select(t_dim, 1);
		return t_real * t_real + t_imag * t_imag;
	}

	inline Tensor ones(c10::IntArrayRef size) {
		auto&& real = torch::ones(size, tensor_opt);
		auto&& imag = torch::zeros(size, tensor_opt);
		return torch::stack({ real, imag }, size.size());
	}

	inline Tensor ones_like(const Tensor& tensor) {
		std::vector<int64_t> temp_size(tensor.sizes().begin(), tensor.sizes().end());
		temp_size.erase(temp_size.end() - 1);
		auto&& real = torch::ones(temp_size, tensor_opt);
		auto&& imag = torch::zeros(temp_size, tensor_opt);
		return torch::stack({ real, imag }, temp_size.size());
	}

	inline Tensor zeros(c10::IntArrayRef size) {
		std::vector<int64_t> temp_shape(size.begin(), size.end());
		temp_shape.push_back(2);
		return torch::zeros(temp_shape, tensor_opt);
	}

	inline Tensor zeros_like(const Tensor& tensor) {
		return torch::zeros_like(tensor, tensor_opt);
	}

	Tensor tensordot(const Tensor& a, const Tensor& b,
		c10::IntArrayRef dim_self, c10::IntArrayRef dim_other);

	Tensor einsum(c10::string_view equation, at::TensorList tensors);

	Tensor mul_element_wise(const Tensor& t, Complex s);

	Tensor mul_element_wise(const Tensor& a, const Tensor& b);

	Tensor reciprocal(const Tensor& a);
	
}