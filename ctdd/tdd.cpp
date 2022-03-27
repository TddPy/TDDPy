#include "stdafx.h"
#include "tdd.hpp"
using namespace std;

double weight::EPS = DEFAULT_EPS;

template <>
boost::unordered_set<tdd::TDD<wcomplex>*> tdd::TDD<wcomplex>::m_all_tdds{};

template <>
boost::unordered_set<tdd::TDD<CUDAcpl::Tensor>*> tdd::TDD<CUDAcpl::Tensor>::m_all_tdds{};

template <>
cache::unique_table<wcomplex> node::Node<wcomplex>::m_unique_table{};

template <>
std::shared_mutex node::Node<wcomplex>::unique_table_m{};

template <>
cache::unique_table<CUDAcpl::Tensor> node::Node<CUDAcpl::Tensor>::m_unique_table{};
template <>
std::shared_mutex node::Node<CUDAcpl::Tensor>::unique_table_m{};

template <>
std::pair<std::shared_mutex, cache::CUDAcpl_table<wcomplex>> cache::Global_Cache<wcomplex>::CUDAcpl_cache{};
template <>
std::pair<std::shared_mutex, cache::sum_table<wcomplex>> cache::Global_Cache<wcomplex>::sum_cache{};
template <>
std::pair<std::shared_mutex, cache::trace_table<wcomplex>> cache::Global_Cache<wcomplex>::trace_cache{};

template <>
std::pair<std::shared_mutex, cache::CUDAcpl_table<CUDAcpl::Tensor>> cache::Global_Cache<CUDAcpl::Tensor>::CUDAcpl_cache{};
template <>
std::pair<std::shared_mutex, cache::sum_table<CUDAcpl::Tensor>> cache::Global_Cache<CUDAcpl::Tensor>::sum_cache{};
template <>
std::pair<std::shared_mutex, cache::trace_table<CUDAcpl::Tensor>> cache::Global_Cache<CUDAcpl::Tensor>::trace_cache{};


template <>
std::pair<std::shared_mutex, cache::cont_table<wcomplex, wcomplex>> cache::Cont_Cache<wcomplex, wcomplex>::cont_cache{};
template <>
std::pair<std::shared_mutex, cache::cont_table<wcomplex, CUDAcpl::Tensor>> cache::Cont_Cache<wcomplex, CUDAcpl::Tensor>::cont_cache{};
template <>
std::pair<std::shared_mutex, cache::cont_table<CUDAcpl::Tensor, wcomplex>> cache::Cont_Cache<CUDAcpl::Tensor, wcomplex>::cont_cache{};
template <>
std::pair<std::shared_mutex, cache::cont_table<CUDAcpl::Tensor, CUDAcpl::Tensor>> cache::Cont_Cache<CUDAcpl::Tensor, CUDAcpl::Tensor>::cont_cache{};

ThreadPool* wnode::iter_para::p_thread_pool = new ThreadPool(DEFAULT_THREAD_NUM);

template <>
wnode::iter_para::para_coordinator<wcomplex, wcomplex> wnode::iter_para::Para_Crd<wcomplex, wcomplex>::record = wnode::iter_para::para_coordinator<wcomplex, wcomplex>();
template <>
wnode::iter_para::para_coordinator<wcomplex, CUDAcpl::Tensor> wnode::iter_para::Para_Crd<wcomplex, CUDAcpl::Tensor>::record = wnode::iter_para::para_coordinator<wcomplex, CUDAcpl::Tensor>();
template <>
wnode::iter_para::para_coordinator<CUDAcpl::Tensor, wcomplex> wnode::iter_para::Para_Crd<CUDAcpl::Tensor, wcomplex>::record = wnode::iter_para::para_coordinator<CUDAcpl::Tensor, wcomplex>();
template <>
wnode::iter_para::para_coordinator<CUDAcpl::Tensor, CUDAcpl::Tensor> wnode::iter_para::Para_Crd<CUDAcpl::Tensor, CUDAcpl::Tensor>::record = wnode::iter_para::para_coordinator<CUDAcpl::Tensor, CUDAcpl::Tensor>();
template <>
shared_mutex wnode::iter_para::Para_Crd<wcomplex, wcomplex>::m{};
template <>
shared_mutex wnode::iter_para::Para_Crd<wcomplex, CUDAcpl::Tensor>::m{};
template <>
shared_mutex wnode::iter_para::Para_Crd<CUDAcpl::Tensor, wcomplex>::m{};
template <>
shared_mutex wnode::iter_para::Para_Crd<CUDAcpl::Tensor, CUDAcpl::Tensor>::m{};

c10::TensorOptions CUDAcpl::tensor_opt = c10::TensorOptions();