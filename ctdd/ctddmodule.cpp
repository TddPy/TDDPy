#include "stdafx.h"
#include "tdd.hpp"

using namespace std;
using namespace node;
using namespace tdd;


/// <summary>
/// delete the tdd passed in (garbage collection)
/// </summary>
/// <typeparam name="W"></typeparam>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
delete_tdd(PyObject* self, PyObject* args) {
	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	TDD<W>* p_tdd = (TDD<W>*)code;
	delete p_tdd;
	return Py_BuildValue("");
}


/// <summary>
/// reset the unique table and all the caches, designate the device.
/// </summary>
/// <param name="self"></param>
/// <param name="args">for storage_order, put in [] from python to indicate the trival order.</param>
/// <returns>the pointer to the tdd</returns>
static PyObject*
reset(PyObject* self, PyObject* args) {
	int device_cuda;
	double new_eps;
	if (!PyArg_ParseTuple(args, "id", &device_cuda, &new_eps))
		return NULL;

	// note that the settings here are shared between scalar and tensor weight.
	TDD<wcomplex>::setting_update(device_cuda, new_eps);
	TDD<wcomplex>::reset();
	TDD<CUDAcpl::Tensor>::reset();
	return Py_BuildValue("");
}


/// <summary>
/// Take in the CUDAcpl tensor, transform to TDD and returns the pointer.
/// </summary>
/// <param name="self"></param>
/// <param name="args">for storage_order, put in [] from python to indicate the trival order.</param>
/// <returns>the pointer to the tdd</returns>
template <class W>
static PyObject*
as_tensor(PyObject* self, PyObject* args)
{
	PyObject* p_tensor, * p_storage_order_ls;
	int dim_parallel;
	if (!PyArg_ParseTuple(args, "OiO", &p_tensor, &dim_parallel, &p_storage_order_ls))
		return NULL;
	auto&& t = THPVariable_Unpack(p_tensor);

	//prepare the storage_order array
	auto&& size = PyList_GET_SIZE(p_storage_order_ls);
	std::vector<int64_t> storage_order(size);
	for (int i = 0; i < size; i++) {
		storage_order[i] = PyLong_AsLongLong(PyList_GetItem(p_storage_order_ls, i));
	}

	//construct the tdd
	auto&& p_res = new TDD<W>(TDD<W>::as_tensor(t, dim_parallel, storage_order));

	// convert to long long
	int64_t code = (int64_t)p_res;
	return Py_BuildValue("L", code);
}

/// <summary>
/// Return the cloned tdd.
/// </summary>
/// <param name="self"></param>
/// <param name="args">for storage_order, put in [] from python to indicate the trival order.</param>
/// <returns>the pointer to the tdd</returns>
template <class W>
static PyObject*
as_tensor_clone(PyObject* self, PyObject* args)
{
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code))
		return NULL;

	TDD<W>* p_tdd = (TDD<W>*)code;

	//construct the tdd
	auto&& p_res = new TDD<W>(*p_tdd);

	// convert to long long
	int64_t res_code = (int64_t)p_res;
	return Py_BuildValue("L", res_code);
}

/// <summary>
/// Return the python torch tensor of the given tdd.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns>The python torch tensor.</returns>
template <class W>
static PyObject*
to_CUDAcpl(PyObject* self, PyObject* args) {

	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	TDD<W>* p_tdd = (TDD<W>*)code;

	auto&& tensor = p_tdd->CUDAcpl();
	return THPVariable_Wrap(tensor);
}


/// <summary>
/// Trace the designated indices of the given tdd.
/// </summary>
/// <typeparam name="W"></typeparam>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
trace(PyObject* self, PyObject* args) {
	int64_t code;
	PyObject* p_i1_pyo, * p_i2_pyo;
	if (!PyArg_ParseTuple(args, "LOO", &code, &p_i1_pyo, &p_i2_pyo)) {
		return NULL;
	}
	TDD<W>* p_tdd = (TDD<W>*)code;

	auto&& size = PyList_GET_SIZE(p_i1_pyo);
	cache::pair_cmd cmd(size);
	for (int i = 0; i < size; i++) {
		cmd[i].first = PyLong_AsLong(PyList_GetItem(p_i1_pyo, i));
		cmd[i].second = PyLong_AsLong(PyList_GetItem(p_i2_pyo, i));
	}

	auto&& p_res = new TDD<W>(p_tdd->trace(cmd));

	// convert to long long
	int64_t code_res = (int64_t)p_res;
	return Py_BuildValue("L", code_res);
}


/// <summary>
/// Return the tensordot of two tdds. The index indication should be a number.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
tensordot_num(PyObject* self, PyObject* args) {
	int64_t code_a, code_b;
	int dim;
	PyObject* p_rearrangement_pyo;
	int parallel_tensor;
	if (!PyArg_ParseTuple(args, "LLiOi", &code_a, &code_b, &dim, &p_rearrangement_pyo, &parallel_tensor)) {
		return NULL;
	}
	TDD<W>* p_tdda = (TDD<W>*)code_a;
	TDD<W>* p_tddb = (TDD<W>*)code_b;

	auto&& size = PyList_GET_SIZE(p_rearrangement_pyo);
	std::vector<int> rearrangement(size);
	for (int i = 0; i < size; i++) {
		rearrangement[i] = PyLong_AsLong(PyList_GetItem(p_rearrangement_pyo, i));
	}

	auto&& p_res = new TDD<W>(TDD<W>::tensordot_num(*p_tdda, *p_tddb, dim, rearrangement, parallel_tensor));
	// convert to long long
	int64_t code = (int64_t)p_res;
	return Py_BuildValue("L", code);
}

/// <summary>
/// Return the tensordot of two tdds. The index indication should be two index lists.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
tensordot_ls(PyObject* self, PyObject* args) {
	int64_t code_a, code_b;
	PyObject* p_i1_pyo, * p_i2_pyo, * p_rearrangement_pyo;
	int parallel_tensor;
	if (!PyArg_ParseTuple(args, "LLOOOi", &code_a, &code_b, &p_i1_pyo, &p_i2_pyo, &p_rearrangement_pyo, &parallel_tensor)) {
		return NULL;
	}
	TDD<W>* p_tdda = (TDD<W>*)code_a;
	TDD<W>* p_tddb = (TDD<W>*)code_b;

	auto size = PyList_GET_SIZE(p_i1_pyo);
	std::vector<int64_t> i1(size);
	std::vector<int64_t> i2(size);
	for (int i = 0; i < size; i++) {
		i1[i] = PyLong_AsLongLong(PyList_GetItem(p_i1_pyo, i));
		i2[i] = PyLong_AsLongLong(PyList_GetItem(p_i2_pyo, i));
	}

	size = PyList_GET_SIZE(p_rearrangement_pyo);
	std::vector<int> rearrangement(size);
	for (int i = 0; i < size; i++) {
		rearrangement[i] = PyLong_AsLong(PyList_GetItem(p_rearrangement_pyo, i));
	}

	auto&& p_res = new TDD<W>(TDD<W>::tensordot(*p_tdda, *p_tddb, i1, i2, rearrangement, parallel_tensor));

	// convert to long long
	int64_t code = (int64_t)p_res;
	return Py_BuildValue("L", code);
}


/// <summary>
/// return the permuted tdd.
/// </summary>
/// <typeparam name="W"></typeparam>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
permute(PyObject* self, PyObject* args) {
	int64_t code;
	PyObject* p_new_order_ls;
	if (!PyArg_ParseTuple(args, "LO", &code, &p_new_order_ls)) {
		return NULL;
	}
	TDD<W>* p_tdd = (TDD<W>*)code;
	auto&& size = PyList_GET_SIZE(p_new_order_ls);
	std::vector<int64_t> new_order(size);
	for (int i = 0; i < size; i++) {
		new_order[i] = PyLong_AsLongLong(PyList_GetItem(p_new_order_ls, i));
	}

	auto&& p_res = new TDD<W>(p_tdd->permute(new_order));

	// convert to long long
	int64_t res_code = (int64_t)p_res;
	return Py_BuildValue("L", res_code);

}




/// <summary>
/// Get the information of a tdd. Return a dictionary.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
get_tdd_info(PyObject* self, PyObject* args) {
	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	TDD<W>* p_tdd = (TDD<W>*)code;

	auto&& tdd_weight = p_tdd->w_node().weight;
	auto&& tdd_node = p_tdd->w_node().node;
	auto&& tdd_dim_parallel = p_tdd->parallel_shape().size();
	auto&& tdd_dim_data = p_tdd->dim_data();
	auto&& tdd_p_parallel_shape = p_tdd->parallel_shape();
	auto&& tdd_p_data_shape = p_tdd->data_shape();
	auto&& tdd_p_storage_order = p_tdd->storage_order();

	// prepare the objects
	auto&& py_weight = THPVariable_Wrap(weight::func<W>::from_weight(tdd_weight));

	auto&& py_parallel_shape = PyTuple_New(tdd_dim_parallel);
	for (int i = 0; i < tdd_dim_parallel; i++) {
		PyTuple_SetItem(py_parallel_shape, i, PyLong_FromLongLong(tdd_p_parallel_shape[i]));
	}

	auto&& py_data_shape = PyTuple_New(tdd_dim_data);
	for (int i = 0; i < tdd_dim_data; i++) {
		PyTuple_SetItem(py_data_shape, i, PyLong_FromLongLong(tdd_p_data_shape[i]));
	}

	auto&& py_storage_order = PyTuple_New(tdd_dim_data);
	for (int i = 0; i < tdd_dim_data; i++) {
		PyTuple_SetItem(py_storage_order, i, PyLong_FromLong(tdd_p_storage_order[i]));
	}

	int64_t py_node_code = (int64_t)tdd_node;

	return Py_BuildValue("{sOsLsisOsisOsO}",
		"weight", py_weight,
		"node", py_node_code,
		"dim parallel", tdd_dim_parallel,
		"parallel shape", py_parallel_shape,
		"dim data", tdd_dim_data,
		"data shape", py_data_shape,
		"storage order", py_storage_order
	);
}

/// <summary>
/// Get the size (non-terminal nodes) of the tdd.
/// </summary>
/// <typeparam name="W"></typeparam>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
get_tdd_size(PyObject* self, PyObject* args) {
	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	TDD<W>* p_tdd = (TDD<W>*)code;
	auto&& size = p_tdd->size();

	return PyLong_FromLong(size);
}

/// <summary>
/// Get the information of a node. Return a dictionary.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
template <class W>
static PyObject*
get_node_info(PyObject* self, PyObject* args) {
	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	const Node<W>* p_node = (const Node<W>*)code;

	auto&& node_id = p_node->get_id();
	auto&& node_order = p_node->get_order();
	auto&& node_range = p_node->get_range();

	auto&& node_successors = p_node->get_successors();

	auto&& py_successors = PyTuple_New(node_range);
	for (int i = 0; i < node_range; i++) {
		auto&& temp_succ = Py_BuildValue("{sOsO}",
			"weight", THPVariable_Wrap(weight::func<W>::from_weight(node_successors[i].weight)),
			"node", PyLong_FromLongLong((int64_t)node_successors[i].node));
		
		PyTuple_SetItem(py_successors, i, temp_succ);
	}

	return Py_BuildValue("{sisisisO}",
		"id", node_id,
		"order", node_order,
		"range", node_range,
		"successors", py_successors
	);
}




static PyMethodDef ctdd_methods[] = {
	{ "delete_tdd", (PyCFunction)delete_tdd<wcomplex>, METH_VARARGS, "delete the tdd passed in (garbage collection)" },
	{ "delete_tdd_T", (PyCFunction)delete_tdd<CUDAcpl::Tensor>, METH_VARARGS, "delete the tdd passed in (garbage collection)" },
	{ "reset", (PyCFunction)reset, METH_VARARGS, " reset the unique table and all the caches, designate the device." },
	{ "as_tensor", (PyCFunction)as_tensor<wcomplex>, METH_VARARGS, "Take in the CUDAcpl tensor, transform to TDD and returns the pointer." },
	{ "as_tensor_T", (PyCFunction)as_tensor<CUDAcpl::Tensor>, METH_VARARGS, "Take in the CUDAcpl tensor, transform to TDD and returns the pointer." },
	{ "as_tensor_clone", (PyCFunction)as_tensor_clone<wcomplex>, METH_VARARGS, "Return the cloned tdd." },
	{ "as_tensor_clone_T", (PyCFunction)as_tensor_clone<CUDAcpl::Tensor>, METH_VARARGS, "Return the cloned tdd." },
	{ "to_CUDAcpl", (PyCFunction)to_CUDAcpl<wcomplex>, METH_VARARGS, "Return the python torch tensor of the given tdd." },
	{ "to_CUDAcpl_T", (PyCFunction)to_CUDAcpl<CUDAcpl::Tensor>, METH_VARARGS, "Return the python torch tensor of the given tdd." },
	{ "trace", (PyCFunction)trace<wcomplex>, METH_VARARGS, "Trace the designated indices of the given tdd." },
	{ "trace_T", (PyCFunction)trace<CUDAcpl::Tensor>, METH_VARARGS, "Trace the designated indices of the given tdd." },
	{ "tensordot_num", (PyCFunction)tensordot_num<wcomplex>, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be a number." },
	{ "tensordot_num_T", (PyCFunction)tensordot_num<CUDAcpl::Tensor>, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be a number." },
	{ "tensordot_ls", (PyCFunction)tensordot_ls<wcomplex>, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be two index lists." },
	{ "tensordot_ls_T", (PyCFunction)tensordot_ls<CUDAcpl::Tensor>, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be two index lists." },
	{ "permute", (PyCFunction)permute<wcomplex>, METH_VARARGS, "return the permuted tdd." },
	{ "permute_T", (PyCFunction)permute<CUDAcpl::Tensor>, METH_VARARGS, "return the permuted tdd." },
	{ "get_tdd_info", (PyCFunction)get_tdd_info<wcomplex>, METH_VARARGS, "Get the information of a tdd. Return a dictionary." },
	{ "get_tdd_info_T", (PyCFunction)get_tdd_info<CUDAcpl::Tensor>, METH_VARARGS, "Get the information of a tdd. Return a dictionary." },
	{ "get_tdd_size", (PyCFunction)get_tdd_size<wcomplex>, METH_VARARGS, "Get the size (non-terminal nodes) of the tdd." },
	{ "get_tdd_size_T", (PyCFunction)get_tdd_size<CUDAcpl::Tensor>, METH_VARARGS, "Get the size (non-terminal nodes) of the tdd." },
	{ "get_node_info", (PyCFunction)get_node_info<wcomplex>, METH_VARARGS, "Get the information of a node. Return a dictionary." },
	{ "get_node_info_T", (PyCFunction)get_node_info<CUDAcpl::Tensor>, METH_VARARGS, "Get the information of a node. Return a dictionary." },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef ctdd = {
	PyModuleDef_HEAD_INIT,
	"ctdd",                        // Module name to use with Python import statements
	"The C++ backend of tdd.",  // Module description
	0,
	ctdd_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_ctdd() {
	TDD<wcomplex>::setting_update();
	TDD<wcomplex>::reset();
	TDD<CUDAcpl::Tensor>::reset();
	return PyModule_Create(&ctdd);
}