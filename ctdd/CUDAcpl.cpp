#include "CUDAcpl.h"

using namespace std;
using namespace CUDAcpl;


Complex CUDAcpl::item(const Tensor& t) {
	return Complex(t.index({ "...",0 }).cpu().item().toDouble(),
		t.index({ "...",1 }).cpu().item().toDouble());
}