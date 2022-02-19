#include "tdd.hpp"
#include <time.h>

using namespace std;
using namespace tdd;
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

		auto&& t1 = torch::rand({ 2,3,2,2 });
		//t1[0][0][0] = 1;
		//t1[0][1][0] = 1;
		//t1[1][0][0] = 1;
		//t1[1][1][0] = -1;

		auto&& t2 = torch::rand({ 3,2,2,2 });
		//t2[0][0][0] = 1;
		//t2[0][1][0] = 1;
		//t2[1][0][0] = -1;
		//t2[1][1][0] = 1;



		auto&& res_direct = CUDAcpl::tensordot(t1, t2, { 0 ,1 }, { 2, 0 });
		//auto res_direct = t1 + t2;
		//std::cout << res_direct << std::endl;


		auto&& t1_tdd = TDD<wcomplex>::as_tensor(t1, 0, {});
		auto&& t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, {});


		//auto res_tdd = TDD<wcomplex>::sum(t1_tdd, t2_tdd);
		auto&& res_tdd = TDD<wcomplex>::tensordot(t1_tdd, t2_tdd, { 0, 1 }, { 2, 0 });

		std::cout << "direct" << std::endl;
		std::cout << res_direct << std::endl;
		std::cout << "tdd" << std::endl;
		std::cout << res_tdd.CUDAcpl() << std::endl;


		auto&& max_diff = torch::max(torch::abs(res_direct - res_tdd.CUDAcpl())).item().toDouble();
		std::cout << "max difference: " << max_diff << endl;
		if (max_diff > 1E-5) {
			cout << "WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
			throw - 100;
		}

		//TDD<wcomplex>::reset();
	}
	double end = clock();

	cout << "total time: " << (end - start) / CLOCKS_PER_SEC << " s" << endl;
	return 0;
}