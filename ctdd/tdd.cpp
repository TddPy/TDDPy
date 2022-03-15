#include "stdafx.h"
#include "tdd.hpp"
using namespace std;

double weight::EPS = DEFAULT_EPS;

int node::Node<wcomplex>::m_global_id = 0;
std::mutex node::Node<wcomplex>::global_id_m{};
int node::Node<CUDAcpl::Tensor>::m_global_id = 0;
std::mutex node::Node<CUDAcpl::Tensor>::global_id_m{};


cache::unique_table<wcomplex>* node::Node<wcomplex>::mp_unique_table = new cache::unique_table<wcomplex>();
std::shared_mutex node::Node<wcomplex>::unique_table_m{};

cache::unique_table<CUDAcpl::Tensor>* node::Node<CUDAcpl::Tensor>::mp_unique_table = new cache::unique_table<CUDAcpl::Tensor>();
std::shared_mutex node::Node<CUDAcpl::Tensor>::unique_table_m{};

cache::CUDAcpl_table<wcomplex>* cache::Global_Cache<wcomplex>::p_CUDAcpl_cache = new cache::CUDAcpl_table<wcomplex>();
cache::sum_table<wcomplex>* cache::Global_Cache<wcomplex>::p_sum_cache = new cache::sum_table<wcomplex>();
std::shared_mutex cache::Global_Cache<wcomplex>::sum_m{};
cache::trace_table<wcomplex>* cache::Global_Cache<wcomplex>::p_trace_cache = new cache::trace_table<wcomplex>();

cache::CUDAcpl_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_CUDAcpl_cache = new cache::CUDAcpl_table<CUDAcpl::Tensor>();
cache::sum_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_sum_cache = new cache::sum_table<CUDAcpl::Tensor>();
std::shared_mutex cache::Global_Cache<CUDAcpl::Tensor>::sum_m{};
cache::trace_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_trace_cache = new cache::trace_table<CUDAcpl::Tensor>();


cache::cont_table<wcomplex, wcomplex>* cache::Cont_Cache<wcomplex, wcomplex>::p_cont_cache = new cache::cont_table<wcomplex, wcomplex>();
cache::cont_table<wcomplex, CUDAcpl::Tensor>* cache::Cont_Cache<wcomplex, CUDAcpl::Tensor>::p_cont_cache = new cache::cont_table<wcomplex, CUDAcpl::Tensor>();
cache::cont_table<CUDAcpl::Tensor, wcomplex>* cache::Cont_Cache<CUDAcpl::Tensor, wcomplex>::p_cont_cache = new cache::cont_table<CUDAcpl::Tensor, wcomplex>();
cache::cont_table<CUDAcpl::Tensor, CUDAcpl::Tensor>* cache::Cont_Cache<CUDAcpl::Tensor, CUDAcpl::Tensor>::p_cont_cache = new cache::cont_table<CUDAcpl::Tensor, CUDAcpl::Tensor>();
shared_mutex cache::Cont_Cache<wcomplex, wcomplex>::m{};
shared_mutex cache::Cont_Cache<wcomplex, CUDAcpl::Tensor>::m{};
shared_mutex cache::Cont_Cache<CUDAcpl::Tensor, wcomplex>::m{};
shared_mutex cache::Cont_Cache<CUDAcpl::Tensor, CUDAcpl::Tensor>::m{};

ThreadPool* wnode::iter_para::p_thread_pool = new ThreadPool(DEFAULT_THREAD_NUM);

wnode::iter_para::para_coordinator<wcomplex, wcomplex> wnode::iter_para::Para_Crd<wcomplex, wcomplex>::record = wnode::iter_para::para_coordinator<wcomplex, wcomplex>();
wnode::iter_para::para_coordinator<wcomplex, CUDAcpl::Tensor> wnode::iter_para::Para_Crd<wcomplex, CUDAcpl::Tensor>::record = wnode::iter_para::para_coordinator<wcomplex, CUDAcpl::Tensor>();
wnode::iter_para::para_coordinator<CUDAcpl::Tensor, wcomplex> wnode::iter_para::Para_Crd<CUDAcpl::Tensor, wcomplex>::record = wnode::iter_para::para_coordinator<CUDAcpl::Tensor, wcomplex>();
wnode::iter_para::para_coordinator<CUDAcpl::Tensor, CUDAcpl::Tensor> wnode::iter_para::Para_Crd<CUDAcpl::Tensor, CUDAcpl::Tensor>::record = wnode::iter_para::para_coordinator<CUDAcpl::Tensor, CUDAcpl::Tensor>();
shared_mutex wnode::iter_para::Para_Crd<wcomplex, wcomplex>::m{};
shared_mutex wnode::iter_para::Para_Crd<wcomplex, CUDAcpl::Tensor>::m{};
shared_mutex wnode::iter_para::Para_Crd<CUDAcpl::Tensor, wcomplex>::m{};
shared_mutex wnode::iter_para::Para_Crd<CUDAcpl::Tensor, CUDAcpl::Tensor>::m{};

c10::TensorOptions CUDAcpl::tensor_opt = c10::TensorOptions();