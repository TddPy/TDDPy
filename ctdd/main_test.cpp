#include "tdd.hpp"

using namespace std;
using namespace tdd;
int main() {

	for (int i = 0; ; i++) {
		cout << "=================== " << i << " ===================" << endl;

		auto t1 = torch::zeros({ 2,2,2 });
		t1[0][0][0] = 1;
		t1[0][1][0] = 1;
		t1[1][0][0] = 1;
		t1[1][1][0] = -1;

		auto t2 = torch::rand({ 2,2,2 });
		t2[0][0][0] = 1;
		t2[0][1][0] = 1;
		t2[1][0][0] = -1;
		t2[1][1][0] = 1;

		auto res_direct = CUDAcpl::tensordot(t1, t2, { 1 }, { 0 });
		//auto res_direct = t1 + t2;
		//std::cout << res_direct << std::endl;


		auto t1_tdd = TDD<wcomplex>::as_tensor(t1, 0, {1, 0});
		auto t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, {0, 1});

		//auto res_tdd = TDD<wcomplex>::sum(t1_tdd, t2_tdd);
		auto res_tdd = TDD<wcomplex>::tensordot(t1_tdd, t2_tdd, { 1 }, { 0 });
		//auto res_tdd = TDD<wcomplex>::direct_product(t1_tdd, t2_tdd);

		std::cout << res_direct << std::endl;
		std::cout << res_tdd.CUDAcpl() << std::endl;

		auto max_diff = torch::max(torch::abs(res_direct - res_tdd.CUDAcpl())).item().toDouble();
		std::cout << "max difference: " << max_diff << endl;
		if (max_diff > 1E-5) {
			cout << "WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
			throw -100;
		}

		//TDD<wcomplex>::reset();
	}
	return 0;
}