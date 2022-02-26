#include "tdd.hpp"
#include <time.h>

using namespace std;
using namespace tdd;

void compare(const torch::Tensor& a, const torch::Tensor& b) {
	auto&& max_diff = torch::max(torch::abs(a - b)).item().toDouble();
	if (max_diff > 1E-7) {
		std::cout << "not passed, max diff: " << max_diff << std::endl;
	}
	else {
		std::cout << "passed, max diff: " << max_diff << std::endl;
	}
}

int main() {


	double start = clock();
	for (int i = 0; i < 100; i++) {
		cout << "=================== " << i << " ===================" << endl;
		/*
		int w = 4;
		std::vector<int64_t> shape(2 * w + 1);
		for (int i = 0; i < 2 * w + 1; i++) {
			shape[i] = 2;
		}
		std::vector<int64_t> i1(w);
		std::vector<int64_t> i2(w);
		for (int i = 0; i < w; i++) {
			i1[i] = 2 * i;
			i2[i] = 2 * i + 1;
		}
		*/

		auto&& sigmax = torch::tensor({ 0.,0.,1.,0.,1.,0.,0.,0. }).reshape({ 2,2,2 });
		auto&& sigmay = torch::tensor({ 0.,0.,0.,-1.,0.,1.,0.,0. }).reshape({ 2,2,2 });
		auto&& hadamard = torch::tensor({ 1.,0.,1.,0.,1.,0.,-1.,0. }).reshape({ 2,2,2 }) / sqrt(2);
		auto&& cnot = torch::tensor({ 1., 0., 0., 0., 0., 0., 0., 0.,
									 0., 0., 1., 0., 0., 0., 0., 0.,
									 0., 0., 0., 0., 0., 0., 1., 0.,
									 0., 0., 0., 0., 1., 0., 0., 0. }).reshape({ 2,2,2,2,2 });

		auto&& cz = torch::tensor({ 1., 0., 0., 0., 0., 0., 0., 0.,
									 0., 0., 1., 0., 0., 0., 0., 0.,
									 0., 0., 0., 0., 1., 0., 0., 0.,
									 0., 0., 0., 0., 0., 0., 0., 1. }).reshape({ 2,2,2,2,2 });
		auto&& I = torch::tensor({ 1., 0., 0., 0., 0., 0., 1., 0. }).reshape({ 2,2,2 });


		auto&& tdd_I = TDD<wcomplex>::as_tensor(I, 0, { 0,1 });
		auto&& tdd_hadamard = TDD<wcomplex>::as_tensor(hadamard, 0, { 0,1 });
		auto&& tdd_cz = TDD<wcomplex>::as_tensor(cz, 0, { 0,2,1,3 });


		auto&& t0 = CUDAcpl::tensordot(hadamard, I, { 0 }, { 1 });
		auto&& tdd_t0 = TDD<wcomplex>::tensordot(tdd_hadamard, tdd_I, { 0 }, { 1 }, { false, true });
		compare(t0, tdd_t0.CUDAcpl());

		// step 1
		auto&& t1 = CUDAcpl::tensordot(hadamard, I, { 0 }, { 1 });
		auto&& tdd_t1 = TDD<wcomplex>::tensordot(tdd_hadamard, tdd_I, { 0 }, { 1 }, { false, true });
		compare(t1, tdd_t1.CUDAcpl());

		// step 2
		auto&& t2 = CUDAcpl::tensordot(I, cz, { 1 }, { 1 });
		auto&& tdd_t2 = TDD<wcomplex>::tensordot(tdd_I, tdd_cz, { 1 }, { 1 }, { false, true, false, false });
		compare(t2, tdd_t2.CUDAcpl());

		// step 3
		auto&& t3 = CUDAcpl::tensordot(hadamard, t2, { 0 }, { 3 });
		auto&& tdd_t3 = TDD<wcomplex>::tensordot(tdd_hadamard, tdd_t2, { 0 }, { 3 }, { false, false, false ,true });
		compare(t3, tdd_t3.CUDAcpl());

		// step 4
		auto&& t4 = CUDAcpl::tensordot(t1, t3, { 0 }, { 2 });
		auto&& tdd_t4 = TDD<wcomplex>::tensordot(tdd_t1, tdd_t3, { 0 }, { 2 }, { true, false, false, false });
		compare(t4, tdd_t4.CUDAcpl());



		//TDD<wcomplex>::reset();
	}
	double end = clock();

	cout << "total time: " << (end - start) / CLOCKS_PER_SEC << " s" << endl;
	return 0;
}