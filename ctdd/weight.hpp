#pragma once
#include "stdafx.h"

namespace weight {
	extern double EPS;

	typedef int64_t WCode;

	// template metaprogramming, return the weight type of combined cont with W1 and W2
	template <typename W1, typename W2> class W_ {};

	template <>
	class W_ <wcomplex, wcomplex> { public:  typedef wcomplex reType; };
	template <>
	class W_ <wcomplex, CUDAcpl::Tensor> { public: typedef CUDAcpl::Tensor reType; };
	template <>
	class W_<CUDAcpl::Tensor, wcomplex> { public: typedef CUDAcpl::Tensor reType; };
	template <>
	class W_<CUDAcpl::Tensor, CUDAcpl::Tensor> { public: typedef CUDAcpl::Tensor reType; };

	template <typename W1, typename W2>
	using W_C = typename W_<W1, W2>::reType;

	inline void get_int_key(WCode* p_vec, double weight) noexcept {
		*p_vec = (WCode)round(weight / EPS);
	}

	inline void get_int_key(WCode* p_vec, CUDAcpl::Tensor weight) {
		auto&& temp = torch::round(weight / EPS).toType(c10::ScalarType::Long).cpu();
		auto ptr = (WCode*)temp.data_ptr<WCode>();

		for (int i = 0; i < temp.numel(); i++) {
			p_vec[i] = ptr[i];
		}
	}


	template <class W>
	inline void as_weight(const CUDAcpl::Tensor& t, W& weight, const std::vector<int64_t>& data_shape) noexcept {
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

	/// <summary>
	/// check the relative equality (up the EPS difference)
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns></returns>
	template <class W>
	inline bool is_equal(const W& a, const W& b) noexcept {
		if constexpr (std::is_same_v<W, wcomplex>) {
			auto this_eps = std::norm(a) * EPS;
			return abs(a.real() - b.real()) < this_eps &&
				abs(a.imag() - b.imag()) < this_eps;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			auto this_eps = CUDAcpl::norm(a) * EPS;
			this_eps = torch::stack({ this_eps, this_eps }, this_eps.dim());
			return  torch::all(torch::abs(a - b) < this_eps).item().toBool();
		}
	}

	/// <summary>
	/// check the equality directly, so input a should have already been normalized
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="a"></param>
	/// <returns></returns>
	template <class W>
	inline bool is_zero(const W& a) noexcept {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return abs(a.real()) < EPS && abs(a.imag()) < EPS;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return torch::all(torch::abs(a) < EPS).item().toBool();
		}
	}

	template <class W>
	inline bool is_exact_zero(const W& a) noexcept {
		if constexpr (std::is_same_v<W, wcomplex>) {
			return a.real() == 0. && a.imag() == 0.;
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return torch::all(a == 0.).item().toBool();
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

	template <typename W1, typename W2>
	inline W_C<W1, W2> mul(const W1& a, const W2& b) noexcept {
		return CUDAcpl::mul_element_wise(a, b);
	}

	/// <summary>
	/// get the reciprocal, and if the element in a is zero, the corresponding reciprocal element is zero.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="a"></param>
	/// <returns></returns>
	template <class W>
	inline W reciprocal_without_zero(const W& a) {
		if constexpr (std::is_same_v<W, wcomplex>) {
			if (is_exact_zero(a)) {
				return wcomplex(0., 0.);
			}
			else {
				return wcomplex(1., 0.) / a;
			}
		}
		else if constexpr (std::is_same_v<W, CUDAcpl::Tensor>) {
			return CUDAcpl::reciprocal_without_zero(a);
		}
	}


	template <typename W1, typename W2>
	inline W_C<W1, W2> prepare_weight(const W1& a, const W2& b, bool parallel_tensor) noexcept {
		if constexpr (std::is_same_v<W1, wcomplex> && std::is_same_v<W2, wcomplex>) {
			return a * b;
		}
		else if constexpr (std::is_same_v<W1, CUDAcpl::Tensor> && std::is_same_v<W2, CUDAcpl::Tensor>) {
			if (parallel_tensor) {
				return CUDAcpl::tensordot(a, b, {}, {});
			}
			else {
				return CUDAcpl::mul_element_wise(a, b);
			}
		}
		else {
			return CUDAcpl::mul_element_wise(a, b);
		}
	}

	// expand the dimensions at the back of the tensor weight
	template <typename W1, typename W2>
	inline W_C<W1, W2> weight_expanded_back(const W1& weight, const std::vector<int64_t>& para_shape_res, bool parallel_tensor) {
		if constexpr (std::is_same_v<W1, wcomplex> && std::is_same_v<W2, wcomplex>) {
			return weight;
		}
		else if constexpr (std::is_same_v<W1, CUDAcpl::Tensor> && std::is_same_v<W2, CUDAcpl::Tensor>) {
			if (parallel_tensor) {
				auto&& sizes = weight.sizes();
				auto size = sizes.size();
				std::vector<int64_t> temp_shape(para_shape_res.size() + 1);
				temp_shape[para_shape_res.size()] = 2;
				for (int i = 0; i < size - 1; i++) {
					temp_shape[i] = sizes[i];
				}
				for (int i = size - 1; i < para_shape_res.size(); i++) {
					temp_shape[i] = 1;
				}
				auto res = weight.view(temp_shape);

				for (int i = size - 1; i < para_shape_res.size(); i++) {
					temp_shape[size - 1 + i] = para_shape_res[i];
				}
				return res.expand(temp_shape);

			}
			else {
				return weight;
			}
		}
		else if constexpr (std::is_same_v<W1, CUDAcpl::Tensor> && std::is_same_v<W2, wcomplex>) {
			return weight;
		}
		else if constexpr (std::is_same_v<W1, wcomplex> && std::is_same_v<W2, CUDAcpl::Tensor>) {
			return CUDAcpl::mul_element_wise(CUDAcpl::ones(para_shape_res), weight);
		}
	}

	// expand the dimensions at the front of the tensor weight
	template <typename W1, typename W2>
	inline W_C<W1, W2> weight_expanded_front(const W2& weight, const std::vector<int64_t>& para_shape_res, bool parallel_tensor) {
		if constexpr (std::is_same_v<W1, wcomplex> && std::is_same_v<W2, wcomplex>) {
			return weight;
		}
		else if constexpr (std::is_same_v<W1, CUDAcpl::Tensor> && std::is_same_v<W2, CUDAcpl::Tensor>) {
			if (parallel_tensor) {
				auto&& sizes = weight.sizes();
				auto size = sizes.size();
				std::vector<int64_t> temp_shape(para_shape_res.size() + 1);
				for (int i = 0; i < para_shape_res.size() - size + 1; i++) {
					temp_shape[i] = 1;
				}
				for (int i = 0; i < size; i++) {
					temp_shape[para_shape_res.size() - size + 1 + i] = sizes[i];
				}
				auto res = weight.view(temp_shape);

				for (int i = 0; i < para_shape_res.size() - size + 1; i++) {
					temp_shape[i] = para_shape_res[i];
				}
				return res.expand(temp_shape);

			}
			else {
				return weight;
			}
		}
		else if constexpr (std::is_same_v<W1, CUDAcpl::Tensor> && std::is_same_v<W2, wcomplex>) {
			return CUDAcpl::mul_element_wise(CUDAcpl::ones(para_shape_res), weight);
		}
		else if constexpr (std::is_same_v<W1, wcomplex> && std::is_same_v<W2, CUDAcpl::Tensor>) {
			return weight;
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

		sum_nweights(W&& _nweight1, W&& _nweight2, W&& _renorm_coef) noexcept {
			nweight1 = std::move(_nweight1);
			nweight2 = std::move(_nweight2);
			renorm_coef = std::move(_renorm_coef);
		}
	};
}