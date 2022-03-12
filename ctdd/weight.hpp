#pragma once
#include "stdafx.h"

namespace weight {
	extern double EPS;

	typedef int64_t WCode;

	inline void get_int_key(WCode* p_vec, double weight) {
		*p_vec = (WCode)round(weight / EPS);
	}

	inline void get_int_key(WCode* p_vec, CUDAcpl::Tensor weight) {
		auto&& temp = torch::round(weight / EPS).toType(c10::ScalarType::Long);
		auto ptr = (WCode*)temp.data_ptr();

		for (int i = 0; i < temp.numel(); i++) {
			p_vec[i] = ptr[i];
		}
	}


	template <class W>
	inline void as_weight(const CUDAcpl::Tensor& t, W& weight, const std::vector<int64_t>& data_shape) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			weight = CUDAcpl::item(t);

		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			std::vector<int64_t> temp_shape{ data_shape };
			temp_shape.push_back(2);
			weight = t.view(temp_shape).clone();
		}
	}

	template <class W>
	inline CUDAcpl::Tensor from_weight(const W& weight) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return CUDAcpl::from_complex(weight);
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return weight;
		}
	}

	template <class W>
	inline CUDAcpl::Tensor res_mul_weight(const CUDAcpl::Tensor& tensor, const W& weight) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return CUDAcpl::mul_element_wise(tensor, weight);
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			auto&& sizes = tensor.sizes();
			std::vector<int64_t> temp_shape(sizes.begin(), sizes.end());
			for (int i = weight.dim() - 1; i < temp_shape.size() - 1; i++) {
				temp_shape[i] = 1;
			}
			auto weight_expanded = weight.view(temp_shape);
			return CUDAcpl::mul_element_wise(tensor, weight_expanded.expand_as(tensor));
		}
	}

	template <class W>
	inline bool is_equal(const W& a, const W& b) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return abs(a.real() - b.real()) < EPS && abs(a.imag() - b.imag()) < EPS;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return  torch::all(torch::abs(a - b) < EPS).item().toBool();
		}
	}

	template <class W>
	inline bool is_zero(const W& a) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return abs(a.real()) < EPS && abs(a.imag()) < EPS;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return torch::all(torch::abs(a) < EPS).item().toBool();
		}
	}

	template <class W>
	inline W ones(const std::vector<int64_t>& data_shape) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return wcomplex(1., 0.);
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::ones(data_shape);
		}
	}

	// produce tensor_ones according to parallel_tensor
	template <class W>
	inline W ones_like(const W& weight) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return wcomplex(1., 0.);
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::ones_like(weight);
		}
	}

	template <class W>
	inline W zeros(const std::vector<int64_t>& data_shape) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return wcomplex(0., 0.);
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::zeros(data_shape);
		}
	}

	template <class W>
	inline W zeros_like(const W& weight) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return wcomplex(0., 0.);
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::zeros_like(weight);
		}
	}

	template <class W>
	inline W mul(const W& a, const W& b) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return a * b;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::mul_element_wise(a, b);
		}
	}

	template <class W>
	inline W reciprocal(const W& a) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return wcomplex(1., 0.) / a;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::reciprocal(a);
		}
	}


	template <class W>
	inline W prepare_weight(const W& a, const W& b, bool parallel_tensor) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return a * b;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			if (parallel_tensor) {
				return CUDAcpl::tensordot(a, b, {}, {});
			}
			else {
				return CUDAcpl::mul_element_wise(a, b);
			}
		}
	}

	// expand the dimensions at the back of the tensor weight
	template <class W>
	inline W weight_expanded_back(const W& weight, const std::vector<int64_t>& para_shape_b, bool parallel_tensor) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return weight;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			if (parallel_tensor) {
				auto&& sizes = weight.sizes();
				auto size = sizes.size();
				std::vector<int64_t> temp_shape(size + para_shape_b.size());
				temp_shape[size + para_shape_b.size() - 1] = 2;
				for (int i = 0; i < size - 1; i++) {
					temp_shape[i] = sizes[i];
				}
				for (int i = 0; i < para_shape_b.size(); i++) {
					temp_shape[size - 1 + i] = 1;
				}
				auto res = weight.view(temp_shape);

				for (int i = 0; i < para_shape_b.size(); i++) {
					temp_shape[size - 1 + i] = para_shape_b[i];
				}
				return res.expand(temp_shape);

			}
			else {
				return weight;
			}
		}
	}

	// expand the dimensions at the front of the tensor weight
	template <class W>
	inline W weight_expanded_front(const W& weight, const std::vector<int64_t>& para_shape_a, bool parallel_tensor) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return weight;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			if (parallel_tensor) {
				auto&& sizes = weight.sizes();
				auto size = sizes.size();
				std::vector<int64_t> temp_shape(size + para_shape_a.size());
				for (int i = 0; i < para_shape_a.size(); i++) {
					temp_shape[i] = 1;
				}
				for (int i = 0; i < size; i++) {
					temp_shape[para_shape_a.size() + i] = sizes[i];
				}
				auto res = weight.view(temp_shape);

				for (int i = 0; i < para_shape_a.size(); i++) {
					temp_shape[i] = para_shape_a[i];
				}
				return res.expand(temp_shape);

			}
			else {
				return weight;
			}
		}
	}


	/// <summary>
	/// structure of normalized sum weights
	/// </summary>
	/// <typeparam name="W"></typeparam>
	template <class W>
	struct sum_nweights {
		W nweight1;
		W nweight2;
		W renorm_coef;

		sum_nweights(W&& _nweight1, W&& _nweight2, W&& _renorm_coef) {
			nweight1 = std::move(_nweight1);
			nweight2 = std::move(_nweight2);
			renorm_coef = std::move(_renorm_coef);
		}
	};
}