#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace node {
	template <typename W>
	class Node;
}

namespace tdd {
	template <class W>
	class TDD;
}

namespace Ctdd {

	// two data types
	typedef std::complex<double> wcomplex;
	typedef torch::Tensor Tensor;


	// template metaprogramming, return the weight type of combined cont with W1 and W2
	template <typename W1, typename W2> class W_ {};

	template <>
	class W_ <wcomplex, wcomplex> { public:  typedef wcomplex reType; };
	template <>
	class W_ <wcomplex, Tensor> { public: typedef Tensor reType; };
	template <>
	class W_<Tensor, wcomplex> { public: typedef Tensor reType; };
	template <>
	class W_<Tensor, Tensor> { public: typedef Tensor reType; };

	template <typename W1, typename W2>
	using W_C = typename W_<W1, W2>::reType;



	/// <summary>
	/// initialize the module
	/// </summary>
	void init();

	
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
	/// <typeparam name="W"></typeparam>
	/// <param name="tensor"></param>
	/// <param name="dim_parallel"></param>
	/// <param name="storage_order"></param>
	/// <returns></returns>
	template <class W>
	tdd::TDD<W>* as_tensor(Tensor tensor, int dim_parallel, std::vector<int64_t> storage_order) {
		return new tdd::TDD<W>(tdd::TDD<W>::as_tensor(tensor, dim_parallel, storage_order));
	}

	/// <summary>
	/// Return the cloned tdd.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	template <class W>
	tdd::TDD<W>* as_tensor_clone(tdd::TDD<W>* p_tdd) {
		return new tdd::TDD<W>(*p_tdd);
	}

	/// <summary>
	/// Return the python torch tensor of the given tdd.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	template <class W>
	Tensor to_CUDAcpl(tdd::TDD<W>* p_tdd) {
		return p_tdd->CUDAcpl();
	}

	/// <summary>
	/// Return the sum of the two tdds.
	/// </summary>
	template <class W>
	tdd::TDD<W>* sum(tdd::TDD<W>* p_tdda, tdd::TDD<W>* p_tddb) {
		return new tdd::TDD<W>(tdd::TDD<W>::sum(*p_tdda, p_tddb));
	}

	/// <summary>
	/// Trace the designated indices of the given tdd.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <param name="indices1"></param>
	/// <param name="indices2"></param>
	/// <returns></returns>
	template <class W>
	tdd::TDD<W>* trace(tdd::TDD<W>* p_tdd, std::vector<int64_t> indices1, std::vector<int64_t> indices2) {
		auto&& size = indices1.size();
		std::vector<std::pair<int64_t, int64_t>> cmd(size);
		for (int i = 0; i < size; i++) {
			cmd[i].first = indices1[i];
			cmd[i].second = indices2[i];
		}
		return new tdd::TDD<W>(p_tdd->trace(cmd));
	}

	/// <summary>
	/// Return the tensordot of two tdds. The index indication should be a number.
	/// </summary>
	/// <typeparam name="W1"></typeparam>
	/// <typeparam name="W2"></typeparam>
	/// <param name="p_tdda"></param>
	/// <param name="p_tddb"></param>
	/// <param name="dim"></param>
	/// <param name="rearrangement"></param>
	/// <param name="parallel_tensor"></param>
	/// <returns></returns>
	template <class W1, class W2>
	tdd::TDD<W_C<W1, W2>>* tensordot_num(
		tdd::TDD<W1>* p_tdda, tdd::TDD<W2>* p_tddb,
		int dim, std::vector<int64_t> rearrangement = {}, bool parallel_tensor = false) {
		return new tdd::TDD<W_C<W1, W2>>(tdd::tensordot_num<W1, W2>(*p_tdda, *p_tddb, dim, rearrangement, parallel_tensor));
	}

	/// <summary>
	/// Return the tensordot of two tdds. The index indication should be two index lists.
	/// </summary>
	/// <typeparam name="W1"></typeparam>
	/// <typeparam name="W2"></typeparam>
	/// <param name="p_tdda"></param>
	/// <param name="p_tddb"></param>
	/// <param name="ils_a"></param>
	/// <param name="ils_b"></param>
	/// <param name="rearrangement"></param>
	/// <param name="parallel_tensor"></param>
	/// <returns></returns>
	template <class W1, class W2>
	tdd::TDD<W_C<W1, W2>>* tensordot_ls(
		tdd::TDD<W1>* p_tdda, tdd::TDD<W2>* p_tddb,
		std::vector<int64_t> ils_a, std::vector<int64_t> ils_b,
		std::vector<int64_t> rearrangement = {}, bool parallel_tensor = false) {
		return new tdd::TDD<W_C<W1, W2>>(tdd::tensordot<W1, W2>(*p_tdda, *p_tddb, ils_a, ils_b, rearrangement, parallel_tensor));
	}

	/// <summary>
	/// return the permuted tdd.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <param name="permutation"></param>
	/// <returns></returns>
	template <class W>
	tdd::TDD<W>* permute(tdd::TDD<W>* p_tdd, std::vector<int64_t> permutation) {
		return new tdd::TDD<W>(p_tdd->permute(permutation));
	}

	/// <summary>
	/// Return the conjugate of the tdd.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	template <class W>
	tdd::TDD<W>* conj(tdd::TDD<W>* p_tdd) {
		return new tdd::TDD<W>(p_tdd->conj());
	}

	/// <summary>
	/// Return the tdd multiplied by the scalar.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <param name="scalar"></param>
	/// <returns></returns>
	template <class W>
	tdd::TDD<W>* mul__w(tdd::TDD<W>* p_tdd, wcomplex scalar) {
		return new tdd::TDD<W>(tdd::operator*(*p_tdd, scalar));
	}

	/// <summary>
	/// Return the tdd multiplied by the tensor (element wise).
	/// </summary>
	/// <param name="p_tdd"></param>
	/// <param name="tensor"></param>
	/// <returns></returns>
	tdd::TDD<Tensor>* mul_tt(tdd::TDD<Tensor>* p_tdd, Tensor tensor);

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
	/// <typeparam name="W"></typeparam>
	/// <param name="p_tdd"></param>
	/// <returns></returns>
	template <class W>
	TDDInfo get_tdd_info(tdd::TDD<W>* p_tdd) {
		TDDInfo res;
		res.weight = p_tdd->w_node().weight;
		res.tdd_node = p_tdd->w_node().get_node();
		res.tdd_dim_parallel = p_tdd->parallel_shape().size();
		res.tdd_dim_data = p_tdd->dim_data();
		res.parallel_shape = p_tdd->parallel_shape();
		res.data_shape = p_tdd->data_shape();
		res.storage_order = p_tdd->storage_order();

		return res;
	}
	

	/// <summary>
	/// Get the size (non-terminal nodes) of the tdd.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	/// <param name=""></param>
	/// <returns></returns>
	template <class W>
	long get_tdd_size(tdd::TDD<W>* p_tdd) {
		return p_tdd->size();
	}
}