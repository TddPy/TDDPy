#include "weight.h"
using namespace weight;

weight::sum_nweights<wcomplex>::sum_nweights(wcomplex&& _nweight1, wcomplex&& _nweight2, wcomplex&& _renorm_coef) {
	nweight1 = std::move(_nweight1);
	nweight2 = std::move(_nweight2);
	renorm_coef = std::move(_renorm_coef);
}
