/*

	The operations on weights in diagrams.

	We are not using class instances as weights because the virtual
	function table will bring along extra memory overhead.

	We use templates to manage the different implementations of the same
	operation in the logic level.

*/

#pragma once
#include <boost\unordered_map.hpp>

namespace weight {


	template <class T_WEIGHT>
	inline std::size_t get_key(T_WEIGHT weight) noexcept {
		throw -1;
	}

	template <class T_WEIGHT, class T_TENSOR>
	inline T_WEIGHT element_to_weight(const T_TENSOR& element) noexcept {
		throw -1;
	}

	template <class T_WEIGHT, class T_TENSOR>
	inline T_TENSOR weight_to_element(const T_WEIGHT& weight) noexcept {
		throw -1;
	}

	template <class T_WEIGHT>
	inline bool is_close(const T_WEIGHT& a, const T_WEIGHT& b) noexcept	{
		throw -1;
	}

	template <class T_WEIGHT>
	inline T_WEIGHT mul(const T_WEIGHT& a, const T_WEIGHT& b) noexcept {
		throw -1;
	}

	template <class T_WEIGHT>
	inline bool is_exact_zero(const T_WEIGHT& a) noexcept {
		throw -1;
	}

	template <class T_WEIGHT>
	inline bool is_zero(const T_WEIGHT& a) noexcept {
		throw -1;
	}

	template <class T_WEIGHT>
	inline T_WEIGHT zeros() noexcept {
		throw -1;
	}

	template <class T_WEIGHT>
	inline T_WEIGHT ones() noexcept {
		throw -1;
	}
	

	template <class T_WEIGHT>
	inline T_WEIGHT reciprocal_without_zero(const T_WEIGHT& a) noexcept {
		throw -1;
	}

}