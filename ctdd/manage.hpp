#pragma once
#include "stdafx.h"
#include "tdd.hpp"

namespace mng {

	inline void print_resource_state() {
		std::cout << std::endl;
		auto all_tdds_w = tdd::TDD<wcomplex>::get_all_tdds();
		std::cout << "wcomplex tdd number: " << all_tdds_w.size() << std::endl;
		auto table_w = node::Node<wcomplex>::get_unique_table();
		int ref_max = 0, ref_min = (std::numeric_limits<int>::max)();
		int zero_ref_count = 0;
		for (auto&& pair : table_w) {
			auto count = pair.second->get_ref_count();
			ref_max = ref_max <= count ? count : ref_max;
			ref_min = ref_min >= count ? count : ref_min;
			if (count == 0) zero_ref_count++;
		}
		std::cout << "wcomplex node number: " << table_w.size() << std::endl;
		std::cout << "wcomplex max reference: " << ref_max << std::endl;
		std::cout << "wcomplex min reference: " << ref_min << std::endl;
		std::cout << "wcomplex 0-ref node number: " << zero_ref_count << std::endl;



		auto all_tdds_t = tdd::TDD<CUDAcpl::Tensor>::get_all_tdds();
		std::cout << "CUDAcpl::Tensor tdd number: " << all_tdds_t.size() << std::endl;
		auto table_t = node::Node<CUDAcpl::Tensor>::get_unique_table();
		ref_max = 0;
		ref_min = (std::numeric_limits<int>::max)();
		zero_ref_count = 0;
		for (auto&& pair : table_t) {
			auto count = pair.second->get_ref_count();
			ref_max = ref_max <= count ? count : ref_max;
			ref_min = ref_min >= count ? count : ref_min;
			if (count == 0) zero_ref_count++;
		}
		std::cout << "CUDAcpl::Tensor node number: " << table_t.size() << std::endl;
		std::cout << "CUDAcpl::Tensor max reference: " << ref_max << std::endl;
		std::cout << "CUDAcpl::Tensor min reference: " << ref_min << std::endl;
		std::cout << "CUDAcpl::Tensor 0-ref node number: " << zero_ref_count << std::endl;
		std::cout << std::endl;
	}

	template <typename W>
	inline void clear_cache() {
		cache::Global_Cache<W>::CUDAcpl_cache.first.lock();
		cache::Global_Cache<W>::CUDAcpl_cache.second.clear();
		cache::Global_Cache<W>::CUDAcpl_cache.first.unlock();

		cache::Global_Cache<W>::sum_cache.first.lock();
		cache::Global_Cache<W>::sum_cache.second.clear();
		cache::Global_Cache<W>::sum_cache.first.unlock();


		cache::Global_Cache<W>::trace_cache.first.lock();
		cache::Global_Cache<W>::trace_cache.second.clear();
		cache::Global_Cache<W>::trace_cache.first.unlock();

		cache::Cont_Cache<W, wcomplex>::cont_cache.first.lock();
		cache::Cont_Cache<W, wcomplex>::cont_cache.second.clear();
		cache::Cont_Cache<W, wcomplex>::cont_cache.first.unlock();

		cache::Cont_Cache<W, CUDAcpl::Tensor>::cont_cache.first.lock();
		cache::Cont_Cache<W, CUDAcpl::Tensor>::cont_cache.second.clear();
		cache::Cont_Cache<W, CUDAcpl::Tensor>::cont_cache.first.unlock();

		node::Node<W>::clean_unique_table();
	}


	extern uint64_t vmem_limit;

	extern HANDLE current_process;

	extern std::atomic<std::chrono::duration<double>> garbage_check_period;

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
		auto current_vmem = get_vmem();
		if (current_vmem > vmem_limit) {
			clear_cache<wcomplex>();
			clear_cache<CUDAcpl::Tensor>();
		}
	}


	inline void setting_update(int thread_num = DEFAULT_THREAD_NUM,
		bool device_cuda = false, bool double_type = true, double new_eps = DEFAULT_EPS,
		double gc_check_period = DEFAULT_MEM_CHECK_PERIOD, uint64_t vmem_limit_MB = DEFAULT_VMEM_LIMIT / 1024. / 1024.) {

		vmem_limit = vmem_limit_MB * 1024. * 1024.;

		garbage_check_period.store(std::chrono::duration<double>{ gc_check_period });

		delete wnode::iter_para::p_thread_pool;
		wnode::iter_para::p_thread_pool = new ThreadPool(thread_num);

		CUDAcpl::reset(device_cuda, double_type);
		weight::EPS = new_eps;
	}

}