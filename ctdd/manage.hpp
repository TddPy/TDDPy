#pragma once
#include "stdafx.h"
#include "tdd.hpp"

namespace mng {
	inline void setting_update(int thread_num = DEFAULT_THREAD_NUM,
		bool device_cuda = false, bool double_type = true, double new_eps = DEFAULT_EPS) {

		delete wnode::iter_para::p_thread_pool;
		wnode::iter_para::p_thread_pool = new ThreadPool(thread_num);

		CUDAcpl::reset(device_cuda, double_type);
		weight::EPS = new_eps;
	}


	/// <summary>
	/// Note that tdds in tdd_ls have their nodes changed (due to rearrangement of node id)
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="tdd_ls"></param>
	template <typename W>
	inline void reset() {
		tdd::TDD<W>::reset();
		node::Node<W>::reset();
		cache::Global_Cache<W>::p_CUDAcpl_cache->clear();
		cache::Global_Cache<W>::p_sum_cache->clear();
		cache::Global_Cache<W>::p_trace_cache->clear();
		cache::Cont_Cache<W, wcomplex>::p_cont_cache->clear();
		cache::Cont_Cache<W, CUDAcpl::Tensor>::p_cont_cache->clear();
	}

	template <typename W>
	inline void clear_cache() {
		cache::Global_Cache<W>::p_CUDAcpl_cache->clear();

		cache::Global_Cache<W>::sum_m.lock();
		cache::Global_Cache<W>::p_sum_cache->clear();
		cache::Global_Cache<W>::sum_m.unlock();

		cache::Global_Cache<W>::p_trace_cache->clear();

		cache::Cont_Cache<W, wcomplex>::m.lock();
		cache::Cont_Cache<W, wcomplex>::p_cont_cache->clear();
		cache::Cont_Cache<W, wcomplex>::m.unlock();

		cache::Cont_Cache<W, CUDAcpl::Tensor>::m.lock();
		cache::Cont_Cache<W, CUDAcpl::Tensor>::p_cont_cache->clear();
		cache::Cont_Cache<W, CUDAcpl::Tensor>::m.unlock();
	}


	extern uint64_t vmem_limit;

	extern HANDLE current_process;

	extern std::chrono::duration<double> mem_check_period;

	inline void get_current_process() {
		current_process = OpenProcess(PROCESS_ALL_ACCESS, FALSE, getpid());
	}

	/// <summary>
	/// return the virtual memory, in Byte
	/// </summary>
	/// <returns></returns>
	inline uint64_t get_vmem() {
		PROCESS_MEMORY_COUNTERS pmc;
		// get process hanlde by pid
		if (GetProcessMemoryInfo(current_process, &pmc, sizeof(pmc)))
		{
			return pmc.PagefileUsage;
		}
		return 0;
	}


	inline void cache_clear_check() {
		if (get_vmem() > vmem_limit) {
			//clear_cache<wcomplex>();
			//clear_cache<CUDAcpl::Tensor>();
		}
	}


}