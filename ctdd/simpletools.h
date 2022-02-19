#pragma once

#include <iostream>
#include <vector>
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
/// vec should contain at least one element
/// </summary>
/// <returns></returns>
template <typename T>
inline std::pair<int, T> min_iv(const std::vector<T>& vec, bool (*pred)(const T&, const T&)) {
	int i_min = 0;
	T v_min = vec[0];
	for (int i = 1; i < vec.size(); i++) {
		if (pred(vec[i], v_min)) {
			i_min = i;
			v_min = vec[i];
		}
	}
	return std::make_pair(i_min, std::move(v_min));
}

/// <summary>
/// vec should contain at least one element
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="vec"></param>
/// <returns></returns>
template <typename T>
inline std::pair<int, T> min_iv(const std::vector<T>& vec) {
	int i_min = 0;
	T v_min = vec[0];
	for (int i = 1; i < vec.size(); i++) {
		if (vec[i] < v_min) {
			i_min = i;
			v_min = vec[i];
		}
	}
	return std::make_pair(i_min, std::move(v_min));
}

/// <summary>
/// return the newly created memory, with element at pos removed.
/// </summary>
/// <returns></returns>
template <typename T>
inline std::vector<T> removed(const std::vector<T>& vec, int pos) {
	std::vector<T> res = std::vector<T>(vec.size() - 1);
	for (int i = 0; i < pos; i++) {
		res[i] = vec[i];
	}
	for (int i = pos + 1; i < vec.size(); i++) {
		res[i - 1] = vec[i];
	}
	return std::move(res);
}


/// <summary>
/// return the newly created memory, with element val inserted at pos.
/// </summary>
/// <returns></returns>
template <typename T>
inline std::vector<T> inserted(const std::vector<T>& vec, int pos, const T& val) {
	std::vector<T> res = std::vector<T>(vec.size() + 1);
	for (int i = 0; i < pos; i++) {
		res[i] = vec[i];
	}
	res[pos] = val;
	for (int i = pos + 1; i < vec.size() + 1; i++) {
		res[i] = vec[i - 1];
	}
	return std::move(res);
}


/// <summary>
/// print the list
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="p"></param>
/// <param name="size"></param>
template <typename T>
inline void print_ls(const T* p, int size) {
	std::cout << "(";
	for (int i = 0; i < size; i++) {
		std::cout << p[i] << ", ";
	}
	std::cout << ")" << std::endl;
}