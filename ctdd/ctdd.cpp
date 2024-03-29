#include "stdafx.h"
#include "tdd.hpp"
#include "wnode.hpp"
#include "manage.hpp"

#include "ctdd.h"

using namespace std;
using namespace node;
using namespace tdd;
using namespace mng;

namespace Ctdd {
	void init() {
		mng::get_current_process();
		mng::reset();
	}

	c10::TensorOptions get_TensorOption() {
		return CUDAcpl::tensor_opt;
	}

	void test() {
		mng::print_resource_state();
	}

	Config get_config() {
		Config res;

		res.thread_num = wnode::iter_para::p_thread_pool->thread_num();
		res.device_cuda = CUDAcpl::tensor_opt.device_opt() == c10::DeviceType::CUDA;
		res.double_type = CUDAcpl::tensor_opt.dtype_opt() == c10::ScalarType::Double;
		res.eps = weight::EPS;
		res.gc_check_period = mng::garbage_check_period.load().count();
		res.vmem_limit_MB = mng::vmem_limit / 1024. / 1024.;

		return res;
	}

	void delete_tdd(void* p_tdd) {
		delete p_tdd;
	}

	void clear_garbage() {
		mng::clear_garbage();
	}

	void clear_cache() {
		mng::clear_garbage();
		mng::clear_cache();
	}

	void reset(int thread_num = 4, bool device_cuda = false, bool double_type = true,
		double new_eps = DEFAULT_EPS, double gc_check_period = DEFAULT_MEM_CHECK_PERIOD,
		uint64_t vmem_limit_MB = 0Ui64) {
		mng::reset(thread_num, device_cuda, double_type, new_eps, gc_check_period, vmem_limit_MB);
	}

	void* as_tensor(Tensor tensor, int dim_parallel, const std::vector<int64_t>& storage_order) {
		return new TDD<wcomplex>(TDD<wcomplex>::as_tensor(tensor, dim_parallel, storage_order));
	}

	void* as_tensor_clone(void* p_tdd) {
		return new TDD<wcomplex>(*(TDD<wcomplex>*)p_tdd);
	}

	void* storage_calibration_clone(void* p_tdd, const std::vector<int64_t>& new_storage_order) {
		return new TDD<wcomplex>(((TDD<wcomplex>*)p_tdd)->storage_calibration_clone(new_storage_order));
	}


	void* zeros(const std::vector<int64_t>& data_shape, const std::vector<int64_t>& storage_order = {}) {
		return new TDD<wcomplex>(TDD<wcomplex>::zeros({}, data_shape, storage_order));
	}

	void* ones(const std::vector<int64_t>& data_shape, const std::vector<int64_t>& storage_order = {}) {
		return new TDD<wcomplex>(TDD<wcomplex>::ones({}, data_shape, storage_order));
	}


	Tensor to_CUDAcpl(void* p_tdd) {
		return ((TDD<wcomplex>*)p_tdd)->CUDAcpl();
	}

	void* sum(void* p_tdda, void* p_tddb) {
		return new TDD<wcomplex>(tdd::TDD<wcomplex>::sum(*(TDD<wcomplex>*)p_tdda, *(TDD<wcomplex>*)p_tddb));
	}

	void* trace(void* p_tdd, const std::vector<int64_t>& indices1, const std::vector<int64_t>& indices2) {
		auto&& size = indices1.size();
		std::vector<std::pair<int, int>> cmd(size);
		for (int i = 0; i < size; i++) {
			cmd[i].first = indices1[i];
			cmd[i].second = indices2[i];
		}
		return new TDD<wcomplex>(((TDD<wcomplex>*)p_tdd)->trace(cmd));
	}

	void* stack(const std::vector<void*>& tdd_ls) {
		std::vector<TDD<wcomplex>> inst_ls;;
		for (int i = 0; i < tdd_ls.size(); i++) {
			inst_ls.push_back(*((TDD<wcomplex>*)tdd_ls[i]));
		}
		return new TDD<wcomplex>(TDD<wcomplex>::stack(inst_ls));
	}


	void* tensordot_num(
		void* p_tdda, void* p_tddb,
		int dim, const std::vector<int>& rearrangement = {}, bool parallel_tensor = false) {
		return new TDD<wcomplex>(tdd::tensordot_num<wcomplex, wcomplex>(*(TDD<wcomplex>*)p_tdda, *(TDD<wcomplex>*)p_tddb, dim, rearrangement, parallel_tensor));
	}

	void* tensordot_ls(
		void* p_tdda, void* p_tddb,
		const std::vector<int64_t>& ils_a, const std::vector<int64_t>& ils_b,
		const std::vector<int>& rearrangement = {}, bool parallel_tensor = false) {
		return new TDD<wcomplex>(tdd::tensordot<wcomplex, wcomplex>(*(TDD<wcomplex>*)p_tdda, *(TDD<wcomplex>*)p_tddb, ils_a, ils_b, rearrangement, parallel_tensor));
	}

	void* permute(void* p_tdd, const std::vector<int64_t>& permutation) {
		return new TDD<wcomplex>(((TDD<wcomplex>*)p_tdd)->permute(permutation));
	}

	void* conj(void* p_tdd) {
		return new TDD<wcomplex>(((TDD<wcomplex>*)p_tdd)->conj());
	}

	void* norm(void* p_tdd) {
		return new TDD<wcomplex>(((TDD<wcomplex>*)p_tdd)->norm());
	}

	void* mul__w(void* p_tdd, wcomplex scalar) {
		return new TDD<wcomplex>(tdd::operator*(*(TDD<wcomplex>*)p_tdd, scalar));
	}

	TDDInfo get_tdd_info(void* p_tdd) {
		TDDInfo res;
		auto p_tdd_ = (TDD<wcomplex>*)p_tdd;
		res.weight = p_tdd_->w_node().weight;
		res.tdd_node = p_tdd_->w_node().get_node();
		res.tdd_dim_parallel = p_tdd_->parallel_shape().size();
		res.tdd_dim_data = p_tdd_->dim_data();
		res.parallel_shape = p_tdd_->parallel_shape();
		res.data_shape = p_tdd_->data_shape();
		res.storage_order = p_tdd_->storage_order();

		return res;
	}

	long get_tdd_size(void* p_tdd) {
		return ((TDD<wcomplex>*)p_tdd)->size();
	}

	NodeInfo get_node_info(void* p_node) {
		NodeInfo res;
		auto p_node_ = (node::Node<wcomplex>*)p_node;
		res.order = p_node_->get_order();
		res.range = p_node_->get_range();
		res.ref_count = p_node_->get_ref_count();

		auto&& successors = p_node_->get_successors();
		res.succ_weights.resize(res.range);
		res.succ_nodes.resize(res.range);
		for (int i = 0; i < res.range; i++) {
			res.succ_weights[i] = successors[i].weight;
			res.succ_nodes[i] = successors[i].get_node();
		}
		return res;
	}


}