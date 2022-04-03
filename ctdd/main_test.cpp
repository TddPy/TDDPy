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

	reset(1);

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


	int range = 2;
	int count = 10;
	auto t1 = torch::rand({ count, range,range,range,range,range,2 }, CUDAcpl::tensor_opt);
	auto t2 = torch::rand({ count*2, range,range,range,range,range,2 }, CUDAcpl::tensor_opt);
	auto t1_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t1, 1, {});
	auto t2_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t2, 1, {});

	auto start = clock();
	for (int i = 0; i < 1; i++) {
		cout << "=================== " << i << " ===================" << endl;
		mng::clear_cache<CUDAcpl::Tensor>();
		auto tdd_res = tdd::tensordot(t1_tdd, t2_tdd, { 1,2 }, { 0,2 }, {}, true);
		//auto actual = tdd_res.CUDAcpl();

	}
	auto end = clock();
	cout << "total time: " << (double)(end - start) / CLOCKS_PER_SEC << " s" << endl;

	delete wnode::iter_para::p_thread_pool;
	return 0;
}