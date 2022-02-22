#include "tdd.hpp"

double weight::EPS = DEFAULT_EPS;

int node::Node<wcomplex>::m_global_id = 0;

cache::unique_table<wcomplex> node::Node<wcomplex>::m_unique_table = cache::unique_table<wcomplex>();

cache::duplicate_table<wcomplex>* cache::Global_Cache<wcomplex>::p_duplicate_cache = new cache::duplicate_table<wcomplex>();
cache::append_table<wcomplex>* cache::Global_Cache<wcomplex>::p_append_cache = new cache::append_table<wcomplex>();
cache::CUDAcpl_table<wcomplex>* cache::Global_Cache<wcomplex>::p_CUDAcpl_cache = new cache::CUDAcpl_table<wcomplex>();
cache::sum_table<wcomplex>* cache::Global_Cache<wcomplex>::p_sum_cache = new cache::sum_table<wcomplex>();
cache::trace_table<wcomplex>* cache::Global_Cache<wcomplex>::p_trace_cache = new cache::trace_table<wcomplex>();

c10::TensorOptions CUDAcpl::tensor_opt = c10::TensorOptions();