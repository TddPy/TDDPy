#include "tdd.hpp"
#include "manage.hpp"
#include <time.h>
#include "ThreadPool.h"

using namespace std;
using namespace tdd;
using namespace mng;

void compare(const torch::Tensor& a, const torch::Tensor& b) {
	auto&& max_diff = (torch::max)(torch::abs(a - b)).item().toDouble();
	if (max_diff > 1E-7) {
		std::cout << "not passed, max diff: " << max_diff << std::endl;
	}
	else {
		std::cout << "passed, max diff: " << max_diff << std::endl;
	}
}

int main() {

	setting_update(1, 0, 1, 1E-14);
	
	auto&& sigmax = torch::tensor({ 0.,0.,1.,0.,1.,0.,0.,0. }, CUDAcpl::tensor_opt).reshape({ 2,2,2 });
	auto&& sigmay = torch::tensor({ 0.,0.,0.,-1.,0.,1.,0.,0. }, CUDAcpl::tensor_opt).reshape({ 2,2,2 });
	auto&& hadamard = torch::tensor({ 1.,0.,1.,0.,1.,0.,-1.,0. }, CUDAcpl::tensor_opt).reshape({ 2,2,2 }) / sqrt(2);
	auto&& cnot = torch::tensor({ 1., 0., 0., 0., 0., 0., 0., 0.,
								 0., 0., 1., 0., 0., 0., 0., 0.,
								 0., 0., 0., 0., 0., 0., 1., 0.,
								 0., 0., 0., 0., 1., 0., 0., 0. }, CUDAcpl::tensor_opt).reshape({ 2,2,2,2,2 });

	auto&& cz = torch::tensor({ 1., 0., 0., 0., 0., 0., 0., 0.,
								 0., 0., 1., 0., 0., 0., 0., 0.,
								 0., 0., 0., 0., 1., 0., 0., 0.,
								 0., 0., 0., 0., 0., 0., -1., 0. }, CUDAcpl::tensor_opt).reshape({ 2,2,2,2,2 });
	auto&& I = torch::tensor({ 1., 0., 0., 0., 0., 0., 1., 0. }, CUDAcpl::tensor_opt).reshape({ 2,2,2 });


	setting_update(4, 0, 1, 1E-14);

	auto start = clock();
	for (int i = 0; i < 10; i++) {
		cout << "=================== " << i << " ===================" << endl;
		
		auto t1 = torch::rand({ 14,2,2,2 }, CUDAcpl::tensor_opt);
		auto t2 = torch::rand({ 14,2,2,2 }, CUDAcpl::tensor_opt);

		auto t1_tdd = TDD<wcomplex>::as_tensor(t1, 0, {});
		auto t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, {});
		auto tdd_res = tdd::tensordot(t1_tdd, t2_tdd, { 1 }, { 1 }, {});
		//auto actual = tdd_res.CUDAcpl();

	}
	auto end = clock();
	cout << "total time: " << (end - start) / CLOCKS_PER_SEC << " s" << endl;
	

	auto cx_tdd = TDD<wcomplex>::as_tensor(cz, 0, { 0,2,1,3 });
	cache::pair_cmd cmd;
	cmd.push_back({ 1, 2 });
	auto res = cx_tdd.trace(cmd);
	cout << res.CUDAcpl() << endl;

	delete wnode::iter_para::p_thread_pool;
	return 0;
}