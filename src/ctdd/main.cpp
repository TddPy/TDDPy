// The main entrance, which provides the debugging of ctdd kernel within C++
//

#include <boost/unordered_map.hpp>
#include <iostream>

using namespace std;

#include "tdd.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"


int main(){
	cout<<"hello"<<endl;
	

	xt::xarray<double> arr1
	{{1.0, 2.0, 3.0},
	{2.0, 5.0, 7.0},
	{2.0, 5.0, 7.0}};

	xt::xarray<double> arr2
	{5.0, 6.0, 7.0};

	xt::xarray<double> res = xt::view(arr1, 1) + arr2;

	std::cout << res;

	xt::xarray<std::complex<double>> arr3
	{5.0, 6.0, 7.0};

	auto test = tdd::TDD<std::complex<double>>::as_tensor(arr3, {});

	return 0;
}
