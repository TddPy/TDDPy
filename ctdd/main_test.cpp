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
	TDD<wcomplex>::setting_update();
	TDD<wcomplex>::reset();
	auto&& sigmax = torch::tensor({ 0.,0.,1.,0.,1.,0.,0.,0. }, c10::ScalarType::Double).reshape({ 2,2,2 });
	auto&& sigmay = torch::tensor({ 0.,0.,0.,-1.,0.,1.,0.,0. }, c10::ScalarType::Double).reshape({ 2,2,2 });
	auto&& hadamard = torch::tensor({ 1.,0.,1.,0.,1.,0.,-1.,0. }, c10::ScalarType::Double).reshape({ 2,2,2 }) / sqrt(2);
	auto&& cnot = torch::tensor({ 1., 0., 0., 0., 0., 0., 0., 0.,
								 0., 0., 1., 0., 0., 0., 0., 0.,
								 0., 0., 0., 0., 0., 0., 1., 0.,
								 0., 0., 0., 0., 1., 0., 0., 0. }, c10::ScalarType::Double).reshape({ 2,2,2,2,2 });

	auto&& cz = torch::tensor({ 1., 0., 0., 0., 0., 0., 0., 0.,
								 0., 0., 1., 0., 0., 0., 0., 0.,
								 0., 0., 0., 0., 1., 0., 0., 0.,
								 0., 0., 0., 0., 0., 0., -1., 0. }, c10::ScalarType::Double).reshape({ 2,2,2,2,2 });
	auto&& I = torch::tensor({ 1., 0., 0., 0., 0., 0., 1., 0. }, c10::ScalarType::Double).reshape({ 2,2,2 });



	double start = clock();
	for (int i = 0; i < 1; i++) {
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


		//auto t1 = CUDAcpl::tensordot(sigmax, sigmay, {}, {});
		auto t1 = cz;
		auto t2 = CUDAcpl::tensordot(sigmax, sigmay, {}, {});
		auto expected = t1 + t2;

		auto t1_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t1, 2, {});
		auto t2_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t2, 2, {});
		auto sum_r = TDD<CUDAcpl::Tensor>::sum(t1_tdd, t2_tdd);
		auto actual = sum_r.CUDAcpl();
		//auto indices = cache::pair_cmd(1);
		//indices[0] = make_pair(0, 2);
		//cout << res.CUDAcpl() << endl;
		compare(t1, t1_tdd.CUDAcpl());
		compare(t2, t2_tdd.CUDAcpl());
		cout << expected << endl;
		cout << actual << endl;

		compare(expected, actual);




		//TDD<wcomplex>::reset();
	}
	double end = clock();

	cout << "total time: " << (end - start) / CLOCKS_PER_SEC << " s" << endl;
	return 0;
}