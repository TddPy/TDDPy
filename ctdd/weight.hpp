#pragma once
#include "stdafx.h"

namespace weight {
	extern double EPS;

	typedef long long WCode;

	inline void get_int_key(WCode* p_vec, double weight) {
		*p_vec = (WCode)round(weight / EPS);
	}

	inline void get_int_key(WCode* p_vec, CUDAcpl::Tensor weight) {
		auto&& temp = torch::round(weight / EPS);
		auto&& ptr = temp.data_ptr<WCode>();

		for (int i = 0; i < temp.numel(); i++) {
			p_vec[i] = ptr[i];
		}
	}

	inline bool is_equal(wcomplex a, wcomplex b) {
		return abs(a.real() - b.real()) < EPS && abs(a.imag() - b.imag()) < EPS;
	}

	inline bool is_zero(wcomplex a) {
		return abs(a.real()) < EPS && abs(a.imag()) < EPS;
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