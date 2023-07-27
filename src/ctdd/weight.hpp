/*

	The operations on weights in diagrams.

*/

#include <boost\unordered_map.hpp>
#pragma once

namespace weight {

	typedef int64_t WCode;

	


	template <class T_WCODE, class T_WEIGHT>
	inline T_WCODE get_key(T_WEIGHT weight){
		throw -1;
	}

	template <class T_WEIGHT, class T_ELEMENT>
	inline T_WEIGHT element_to_weight(T_ELEMENT& element) noexcept {
		throw -1;
	}

	template <class T_WEIGHT, class T_ELEMENT>
	inline T_ELEMENT weight_to_element(T_WEIGHT& weight) noexcept {
		throw -1;
	}
}