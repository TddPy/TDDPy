#include "tdd.hpp"

double weight::EPS = DEFAULT_EPS;

int node::Node<wcomplex>::m_global_id = 0;
int node::Node<CUDAcpl::Tensor>::m_global_id = 0;

cache::unique_table<wcomplex> node::Node<wcomplex>::m_unique_table = cache::unique_table<wcomplex>();
cache::unique_table<CUDAcpl::Tensor> node::Node<CUDAcpl::Tensor>::m_unique_table = cache::unique_table<CUDAcpl::Tensor>();

cache::CUDAcpl_table<wcomplex>* cache::Global_Cache<wcomplex>::p_CUDAcpl_cache = new cache::CUDAcpl_table<wcomplex>();
cache::sum_table<wcomplex>* cache::Global_Cache<wcomplex>::p_sum_cache = new cache::sum_table<wcomplex>();
cache::trace_table<wcomplex>* cache::Global_Cache<wcomplex>::p_trace_cache = new cache::trace_table<wcomplex>();
cache::cont_table<wcomplex>* cache::Global_Cache<wcomplex>::p_cont_cache = new cache::cont_table<wcomplex>();
cache::CUDAcpl_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_CUDAcpl_cache = new cache::CUDAcpl_table<CUDAcpl::Tensor>();
cache::sum_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_sum_cache = new cache::sum_table<CUDAcpl::Tensor>();
cache::trace_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_trace_cache = new cache::trace_table<CUDAcpl::Tensor>();
cache::cont_table<CUDAcpl::Tensor>* cache::Global_Cache<CUDAcpl::Tensor>::p_cont_cache = new cache::cont_table<CUDAcpl::Tensor>();


c10::TensorOptions CUDAcpl::tensor_opt = c10::TensorOptions();