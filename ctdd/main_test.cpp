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
	auto t1 = torch::rand({ count, range, range, 2 }, CUDAcpl::tensor_opt);
	auto t2 = torch::rand({ count*2, range,range,range,range,range,2 }, CUDAcpl::tensor_opt);
	auto t1_tdd = TDD<wcomplex>::as_tensor(t1, 0, {});
	auto t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, {});

	auto t1_indexed = t1.select(0, 0);
	auto t1_indexed_direct = TDD<wcomplex>::as_tensor(t1_indexed, 0, {});
	auto t1_indexed_tdd = t1_tdd.slice({ 0,1 }, { 0,1 });

	std::cout << t1_indexed << endl;
	std::cout << t1_indexed_tdd.CUDAcpl() << endl;

	delete wnode::iter_para::p_thread_pool;
	return 0;
}