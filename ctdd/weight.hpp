#pragma once
#include "stdafx.h"

namespace weight {
	extern double EPS;

	typedef long long WCode;

	inline void get_int_key(WCode* p_vec, double weight) {
		*p_vec = (WCode)round(weight / EPS);
	}

	inline void get_int_key(WCode* p_vec, CUDAcpl::Tensor weight) {
		auto&& temp = torch::round(weight / EPS).toType(c10::ScalarType::Int);
		auto ptr = (WCode*)temp.data_ptr();

		for (int i = 0; i < temp.numel(); i++) {
			p_vec[i] = ptr[i];
		}
	}


	template <class W>
	class func {
	public:
		static inline void as_weight(const CUDAcpl::Tensor& t, W& weight, const std::vector<int64_t>& data_shape);
		static inline CUDAcpl::Tensor from_weight(const W& weight);

		static inline CUDAcpl::Tensor res_mul_weight(const CUDAcpl::Tensor& tensor, const W& weight);

		static inline bool is_equal(const W& a, const W& b);
		static inline bool is_zero(const W& a);

		static inline W ones(const std::vector<int64_t>& data_shape);
		// produce tensor_ones according to parallel_tensor
		static inline W ones_like(const W& weight);
		static inline W zeros(const std::vector<int64_t>& data_shape);
		static inline W zeros_like(const W& weight);

		static inline W mul(const W& a, const W& b);
		static inline W reciprocal(const W& a);


		static inline W prepare_weight(const W& a, const W& b, bool parallel_tensor);

		// expand the dimensions at the back of the tensor weight
		static inline W weight_expanded_back(const W& weight, const std::vector<int64_t>& para_shape_b, bool parallel_tensor);

		// expand the dimensions at the front of the tensor weight
		static inline W weight_expanded_front(const W& weight, const std::vector<int64_t>& para_shape_a, bool parallel_tensor);

	};


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