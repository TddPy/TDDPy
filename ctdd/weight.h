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
}