#pragma once

template <typename T>
inline T* array_clone(const T* p, int size) {
	T* p_res = (T*)malloc(sizeof(T) * size);
	for (int i = 0; i < size; i++) {
		p_res[i] = p[i];
	}
	return p_res;
}

template <typename T>
inline T* array_concat(const T* p_a, int size_a, const T* p_b, int size_b) {
	T* p_res = (T*)malloc(sizeof(T) * (size_a + size_b));
	for (int i = 0; i < size_a; i++) {
		p_res[i] = p_a[i];
	}
	for (int i = 0; i < size_b; i++) {
		p_res[i + size_a] = p_b[i];
	}
	return p_res;
}
