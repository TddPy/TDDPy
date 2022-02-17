#include "tdd.hpp"

using namespace std;
using namespace tdd;
int main() {

	auto t1 = torch::zeros({ 2,2,2,2 });
	for (int i = 0; i < 2; i++) {
		t1[i][0][0][0] = 1;
		t1[i][0][1][0] = 1;
		t1[i][1][0][0] = 1;
		t1[i][1][1][0] = -1;
	}

	//auto t2 = torch::rand({ 3,3,2 });
	//t2[0][0][0] = 1;
	//t2[0][1][0] = 1;
	//t2[1][0][0] = -1;
	//t2[1][1][0] = 1;

	//auto res_direct = CUDAcpl::tensordot(t1, t2, { 1 }, { 0 });
	//auto res_direct = t1 + t2;
	std::cout << t1 << std::endl;

	
	//const int order[] = { 1,0 };
	auto t1_tdd = TDD<wcomplex>::as_tensor(t1, 0, std::vector<int64_t>());

	std::cout << t1_tdd.CUDAcpl() << std::endl;

	return 0;
}