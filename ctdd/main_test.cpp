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
	/*
	int thread_count = 10;

	ThreadPool pool(thread_count);

	std::shared_mutex m;
	int current = 0;

	int count = 1000;

	auto start_ = clock();

	std::vector<std::future<void>> results(thread_count);

	for (int i = 0; i < thread_count; i++) {
		results[i] = pool.enqueue(
			[&] {
				m.lock();
				while (current < count) {
					current++;
					m.unlock();
					double temp = 0.3333;
					for (int j = 0; j < 5000000; j++) {
						temp = temp * temp + temp;
					}
					m.lock();
				}
				m.unlock();
			}
		);
	}	
	for (int i = 0; i < thread_count; i++) {
		results[i].get();
	}
	auto end_ = clock();
	cout << "total time: " << (double)(end_ - start_) / CLOCKS_PER_SEC << " s" << endl;

	std::cout << current << std::endl;

	return 0;
	*/

	reset(4,true);

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
	int count = 1000;
	auto t1 = torch::rand({ count, range,range,range,range,range,2 }, CUDAcpl::tensor_opt);
	auto t2 = torch::rand({ count, range,range,range,range,range,2 }, CUDAcpl::tensor_opt);
	auto t1_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t1, 1, {});
	auto t2_tdd = TDD<CUDAcpl::Tensor>::as_tensor(t2, 1, {});
	for (int ti = 1; ti < 5; ti++) {
		cout << "thread num: " << ti << endl;
		reset(ti,true);
		auto start = clock();
		for (int i = 0; i < 1; i++) {
			cout << "=================== " << i << " ===================" << endl;
			mng::clear_cache<CUDAcpl::Tensor>();
			auto tdd_res = tdd::tensordot(t1_tdd, t2_tdd, { 1,2 }, { 0,2 }, {});
			//auto actual = tdd_res.CUDAcpl();

		}
		auto end = clock();
		cout << "total time: " << (double)(end - start) / CLOCKS_PER_SEC << " s" << endl;
	}
	delete wnode::iter_para::p_thread_pool;
	return 0;
}