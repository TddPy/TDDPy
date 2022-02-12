#pragma once

#include <stdlib.h>
#include <utility>

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

/// <summary>
/// p should contain at least one element
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="p"></param>
/// <param name="size"></param>
/// <returns></returns>
template <typename T>
inline std::pair<int, T> min(const T* p, int size) {
	int index = 0;
	T value = p[0];
	for (int i = 1; i < size; i++) {
		if (p[i] < value) {
			value = p[i];
			index = i;
		}
	}
	return std::make_pair(index, value);
}

/// <summary>
/// return the newly created memory, with element at pos removed.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="p"></param>
/// <param name="size"></param>
/// <param name="pos"></param>
/// <returns></returns>
template <typename T>
inline T* removed(const T* p, int size, int pos) {
	T* p_res = (T*)malloc(sizeof(T) * (size - 1));
	for (int i = 0; i < pos; i++) {
		p_res[i] = p[i];
	}
	for (int i = pos + 1; i < size; i++) {
		p_res[i - 1] = p[i];
	}
	return p_res;
}

/// <summary>
/// return the newly created memory, with element val inserted at pos.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="p"></param>
/// <param name="size"></param>
/// <param name="pos"></param>
/// <param name="val"></param>
/// <returns></returns>
template <typename T>
inline T* inserted(const T* p, int size, int pos, const T& val) {
	T* p_res = (T*)malloc(sizeof(T) * (size + 1));
	for (int i = 0; i < pos; i++) {
		p_res[i] = p[i];
	}
	p_res[pos] = val;
	for (int i = pos + 1; i < size + 1; i++) {
		p_res[i] = p[i - 1];
	}
	return p_res;
}