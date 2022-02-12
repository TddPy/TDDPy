#include "tdd.h"

using namespace std;
using namespace tdd;
int main() {

	auto t1 = torch::zeros({ 2,2,2 });
	t1[0][0][0] = 1;
	t1[0][1][0] = 1;
	t1[1][0][0] = 1;
	t1[1][1][0] = -1;

	auto t2 = torch::zeros({ 2,2,2 });
	t2[0][0][0] = 1;
	t2[0][1][0] = 1;
	t2[1][0][0] = -1;
	t2[1][1][0] = 1;

	auto res_direct = CUDAcpl::tensordot(t1, t2, {0}, {1});
	//auto res_direct = t1 + t2;
	cout << res_direct << endl;

	const int order[] = { 1,0 };
	auto t1_tdd = TDD::as_tensor(t1, 0, order);

	auto t2_tdd = TDD::as_tensor(t2, 0, order);
	//auto res_tdd = TDD::direct_product(t1_tdd, t2_tdd);
	//auto res_tdd = TDD::sum(t1_tdd, t2_tdd);
	auto res_tdd = TDD::tensordot(t1_tdd, t2_tdd, 1);
	res_tdd.__print();
	cout << res_tdd.CUDAcpl() << endl;

	return 0;
}