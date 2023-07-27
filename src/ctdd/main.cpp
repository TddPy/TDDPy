// The main entrance, which provides the debugging of ctdd kernel within C++
//

#include <boost/unordered_map.hpp>
#include <iostream>

using namespace std;

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

// bool operator == (const int& a, const int& b) noexcept{
// 	return a == b;
// }

std::size_t hash_value(const int& key) noexcept{
	return 0;
}


int main(){
	cout<<"hello"<<endl;
	
	#include <iostream>


	xt::xarray<double> arr1
	{{1.0, 2.0, 3.0},
	{2.0, 5.0, 7.0},
	{2.0, 5.0, 7.0}};

	xt::xarray<double> arr2
	{5.0, 6.0, 7.0};

	xt::xarray<double> res = xt::view(arr1, 1) + arr2;

	std::cout << res;
	return 0;
}
