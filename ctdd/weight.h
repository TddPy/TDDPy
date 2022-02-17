#pragma once
#include "stdafx.h"

namespace weight {
	extern double EPS;

	inline int get_int_key(double weight) {
		return (int)round(weight / EPS);
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

		sum_nweights(W&& _nweight1, W&& _nweight2, W&& _renorm_coef);
	};
}