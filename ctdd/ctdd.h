#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace Ctdd {

	// two data types
	typedef std::complex<double> wcomplex;
	typedef torch::Tensor Tensor;


	/// <summary>
	/// initialize the module
	/// </summary>
	void init();

	/// <summary>
	/// get the tensor option used in torch
	/// </summary>
	/// <returns></returns>
	c10::TensorOptions get_TensorOption();

	/// <summary>
	/// this method for testing purpose
	/// </summary>
	void test();

	/// <summary>
	/// describe the configuration of Ctdd
	/// </summary>
	struct Config {
		int thread_num;
		bool device_cuda;
		bool double_type;
		double eps;
		double gc_check_period;
		double vmem_limit_MB;
	};

	/// <summary>
	/// return the current configuration as a struct Config
	/// </summary>
	/// <returns></returns>
	Config get_config();


	/// <summary>
	/// delete the tdd passed in (garbage collection)
	/// </summary>
	/// <param name="p_tdd"></param>
	void delete_tdd(void* p_tdd);


	/// <summary>
	/// clear the garbage only
	/// </summary>
	void clear_garbage();


	/// <summary>
	/// clear all the caches.
	/// </summary>
	void clear_cache();

	/// <summary>
	/// reset the system and update the settings.
	/// </summary>
	/// <param name="thread_num"></param>
	/// <param name="device_cuda"></param>
	/// <param name="double_type"></param>
	/// <param name="new_eps"></param>
	/// <param name="gc_check_period"></param>
	/// <param name="vmem_limit_MB"></param>
	void reset(int thread_num, bool device_cuda, bool double_type,
		double new_eps, double gc_check_period,
		uint64_t vmem_limit_MB);

	/// <summary>
	/// Take in the CUDAcpl tensor (torch::Tensor), transform to TDD and returns the pointer.
	/// </summary>
	/// <param name="tensor"></param>
	/// <param name="dim_parallel"></param>
	/// <param name="storage_order"></param>
	/// <returns></returns>
	void* as_tensor(Tensor tensor, int dim_parallel, std::vector<int64_t> storage_order);

	/// <summary>
	/// Return the cloned tdd.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	void* as_tensor_clone(void* p_tdd);

	/// <summary>
	/// Return the python torch tensor of the given tdd.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	Tensor to_CUDAcpl(void* p_tdd);

	/// <summary>
	/// Return the sum of the two tdds.
	/// </summary>
	void* sum(void* p_tdda, void* p_tddb);

	/// <summary>
	/// Trace the designated indices of the given tdd.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <param name="indices1"></param>
	/// <param name="indices2"></param>
	/// <returns></returns>
	void* trace(void* p_tdd, std::vector<int64_t> indices1, std::vector<int64_t> indices2);

	/// <summary>
	/// Return the tensordot of two tdds. The index indication should be a number.
	/// </summary>
	/// <param name="p_tdda"></param>
	/// <param name="p_tddb"></param>
	/// <param name="dim"></param>
	/// <param name="rearrangement"></param>
	/// <param name="parallel_tensor"></param>
	/// <returns></returns>
	void* tensordot_num(
		void* p_tdda, void* p_tddb,
		int dim, std::vector<int> rearrangement, bool parallel_tensor);

	/// <summary>
	/// Return the tensordot of two tdds. The index indication should be two index lists.
	/// </summary>
	/// <param name="p_tdda"></param>
	/// <param name="p_tddb"></param>
	/// <param name="ils_a"></param>
	/// <param name="ils_b"></param>
	/// <param name="rearrangement"></param>
	/// <param name="parallel_tensor"></param>
	/// <returns></returns>
	void* tensordot_ls(
		void* p_tdda, void* p_tddb,
		std::vector<int64_t> ils_a, std::vector<int64_t> ils_b,
		std::vector<int> rearrangement, bool parallel_tensor);

	/// <summary>
	/// return the permuted tdd.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <param name="permutation"></param>
	/// <returns></returns>
	void* permute(void* p_tdd, std::vector<int64_t> permutation);

	/// <summary>
	/// Return the conjugate of the tdd.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	void* conj(void* p_tdd);

	/// <summary>
	/// Return the tdd multiplied by the scalar.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <param name="scalar"></param>
	/// <returns></returns>
	void* mul__w(void* p_tdd, wcomplex scalar);

	struct TDDInfo {
		wcomplex weight;
		void* tdd_node;
		int tdd_dim_parallel;
		std::vector<int64_t> parallel_shape;
		int tdd_dim_data;
		std::vector<int64_t> data_shape;
		std::vector<int64_t> storage_order;
	};

	/// <summary>
	/// Get the information of a tdd. Return a dictionary.
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	TDDInfo get_tdd_info(void* p_tdd);


	/// <summary>
	/// Get the size (non-terminal nodes) of the tdd.
	/// </summary>
	/// <param name=""></param>
	/// <returns></returns>
	long get_tdd_size(void* p_tdd);
}