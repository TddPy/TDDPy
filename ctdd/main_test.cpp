#include "tdd.h"

using namespace std;
using namespace tdd;
int main() {

	auto t = torch::rand({ 2,2,2 });
	//t[0][0][0] = 1;
	//t[0][1][0] = 1;
	//t[1][0][0] = 1;
	//t[1][1][0] = -1;
	cout << t;
	auto p_res = TDD::as_tensor(t, 0, nullptr);
	cout << p_res->get_size() << endl;
	cout << p_res->CUDAcpl();


	return 0;
}