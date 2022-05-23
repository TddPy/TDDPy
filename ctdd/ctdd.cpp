#include "stdafx.h"
#include "tdd.hpp"
#include "wnode.hpp"
#include "manage.hpp"

using namespace std;
using namespace node;
using namespace tdd;
using namespace mng;

namespace Ctdd {
	void init() {
		get_current_process();
		reset();
	}
	void test() {
		mng::print_resource_state();
	}
}