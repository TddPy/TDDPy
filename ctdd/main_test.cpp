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



	double start = clock();
	

	for (int i = 0; i < 2; i++) {
		cout << "=================== " << i << " ===================" << endl;

		auto t1 = torch::rand({ 2,2,2,2,2,2 }, CUDAcpl::tensor_opt);
		auto t2 = torch::rand({ 2,2,2,2,2,2 }, CUDAcpl::tensor_opt);

		auto t1_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t1, 1, {});
		auto t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, {});
		auto tdd_res = tdd::tensordot<CUDAcpl::Tensor, wcomplex>(t1_tdd, t2_tdd, { 0,1,2 }, { 0,1,2 }, {}, true);
		auto actual = tdd_res.CUDAcpl();
	}
	double end = clock();

	cout << "total time: " << (end - start) / CLOCKS_PER_SEC << " s" << endl;

	
	setting_update(4, 0, 1, 1E-14);

	start = clock();
	for (int i = 0; i < 4; i++) {
		cout << "=================== " << i << " ===================" << endl;
		
		auto t1 = torch::rand({ 2,2,2,2,2,2 }, CUDAcpl::tensor_opt);
		auto t2 = torch::rand({ 2,2,2,2,2,2 }, CUDAcpl::tensor_opt);

		auto t1_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t1, 1, {});
		auto t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, {});
		auto tdd_res = tdd::tensordot<CUDAcpl::Tensor, wcomplex>(t1_tdd, t2_tdd, { 0,1,2 }, { 0,1,2 }, {}, true);
		auto actual = tdd_res.CUDAcpl();

	}
	end = clock();

	cout << "total time: " << (end - start) / CLOCKS_PER_SEC << " s" << endl;
	

	return 0;
}