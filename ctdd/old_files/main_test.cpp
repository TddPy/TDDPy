#include "tdd.h"

using namespace std;
using namespace tdd;
int main() {

	auto t1 = torch::rand({ 2,3,2 });
	//t1[0][0][0] = 1;
	//t1[0][1][0] = 1;
	//t1[1][0][0] = 1;
	//t1[1][1][0] = -1;

	auto t2 = torch::rand({ 3,3,2 });
	//t2[0][0][0] = 1;
	//t2[0][1][0] = 1;
	//t2[1][0][0] = -1;
	//t2[1][1][0] = 1;

	auto res_direct = CUDAcpl::tensordot(t1, t2, { 1 }, { 0 });
	//auto res_direct = t1 + t2;
	std::cout << res_direct << std::endl;

	const int order[] = { 1,0 };
	auto t1_tdd = TDD<wcomplex>::as_tensor(t1, 0, order);

	auto t2_tdd = TDD<wcomplex>::as_tensor(t2, 0, order);
	//auto res_tdd = TDD::direct_product(t1_tdd, t2_tdd);
	//auto res_tdd = TDD::sum(t1_tdd, t2_tdd);
	auto res_tdd = TDD<wcomplex>::tensordot(t1_tdd, t2_tdd, 1);
	res_tdd.__print();
	std::cout << res_tdd.CUDAcpl() << std::endl;

	return 0;
}