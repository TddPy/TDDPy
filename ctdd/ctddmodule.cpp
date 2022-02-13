#include "stdafx.h"
#include "tdd.h"

using namespace std;
using namespace node;
using namespace tdd;


/// <summary>
/// Take in the CUDAcpl tensor, transform to TDD and returns the pointer.
/// </summary>
/// <param name="self"></param>
/// <param name="args">for index_order, put in [] from python to indicate the trival order.</param>
/// <returns>the pointer to the tdd</returns>
static PyObject*
as_tensor(PyObject* self, PyObject* args)
{
	PyObject* p_tensor, * p_index_order_ls;
	int dim_parallel;
	if (!PyArg_ParseTuple(args, "OiO", &p_tensor, &dim_parallel, &p_index_order_ls))
		return NULL;
	auto t = THPVariable_Unpack(p_tensor);

	TDD* p_res;
	//prepare the index_order array
	auto size = PyList_GET_SIZE(p_index_order_ls);
	if (size == 0) {
		p_res = new TDD(TDD::as_tensor(t, dim_parallel, nullptr));
	}
	else {
		int* p_index_order = (int*)malloc(sizeof(int) * size);
		for (int i = 0; i < size; i++) {
			p_index_order[i] = _PyLong_AsInt(PyList_GetItem(p_index_order_ls, i));
		}

		//construct the tdd
		p_res = new TDD(TDD::as_tensor(t, dim_parallel, p_index_order));
		free(p_index_order);
	}

	// convert to long long
	int64_t code = (int64_t)p_res;
	return Py_BuildValue("L", code);
}

/// <summary>
/// Return the python torch tensor of the given tdd.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns>The python torch tensor.</returns>
static PyObject*
to_CUDAcpl(PyObject* self, PyObject* args) {

	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	TDD* p_tdd = (TDD*)code;

	auto&& tensor = p_tdd->CUDAcpl();
	return THPVariable_Wrap(tensor);
}


/// <summary>
/// Return the tensordot of two tdds. The index indication should be a number.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
static PyObject*
tensordot_num(PyObject* self, PyObject* args) {
	int64_t code_a, code_b;
	int dim;
	if (!PyArg_ParseTuple(args, "LLi", &code_a, &code_b, &dim)) {
		return NULL;
	}
	TDD* p_tdda = (TDD*)code_a;
	TDD* p_tddb = (TDD*)code_b;
	auto p_res = new TDD(TDD::tensordot(*p_tdda, *p_tddb, dim));
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
static PyObject*
tensordot_ls(PyObject* self, PyObject* args) {
	int64_t code_a, code_b;
	PyObject* p_i1_pyo, * p_i2_pyo;
	if (!PyArg_ParseTuple(args, "LLOO", &code_a, &code_b, &p_i1_pyo, &p_i2_pyo)) {
		return NULL;
	}
	TDD* p_tdda = (TDD*)code_a;
	TDD* p_tddb = (TDD*)code_b;

	auto size = PyList_GET_SIZE(p_i1_pyo);
	int* p_i1 = (int*)malloc(sizeof(int) * size);
	int* p_i2 = (int*)malloc(sizeof(int) * size);
	for (int i = 0; i < size; i++) {
		p_i1[i] = _PyLong_AsInt(PyList_GetItem(p_i1_pyo, i));
		p_i2[i] = _PyLong_AsInt(PyList_GetItem(p_i2_pyo, i));
	}
	auto p_res = new TDD(TDD::tensordot(*p_tdda, *p_tddb, size, p_i1, p_i2));

	free(p_i1);
	free(p_i2);
	// convert to long long
	int64_t code = (int64_t)p_res;
	return Py_BuildValue("L", code);
}

/// <summary>
/// Get the information of a tdd. Return a dictionary.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
static PyObject*
get_tdd_info(PyObject* self, PyObject* args) {
	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	TDD* p_tdd = (TDD*)code;

	auto tdd_weight = p_tdd->wnode().weight;
	auto tdd_node = p_tdd->wnode().p_node;
	auto tdd_dim_parallel = p_tdd->dim_parallel();
	auto tdd_dim_data = p_tdd->dim_data();
	auto tdd_p_parallel_shape = p_tdd->parallel_shape();
	auto tdd_p_data_shape = p_tdd->data_shape();
	auto tdd_p_index_order = p_tdd->index_order();
	auto tdd_size = p_tdd->get_size();

	// prepare the objects
	auto py_weight = THPVariable_Wrap(CUDAcpl::from_complex(tdd_weight));
	auto py_parallel_shape = PyTuple_New(tdd_dim_parallel);
	for (int i = 0; i < tdd_dim_parallel; i++) {
		PyTuple_SetItem(py_parallel_shape, i, PyLong_FromLongLong(tdd_p_parallel_shape[i]));
	}
	auto py_data_shape = PyTuple_New(tdd_dim_data);
	for (int i = 0; i < tdd_dim_data; i++) {
		PyTuple_SetItem(py_data_shape, i, PyLong_FromLongLong(tdd_p_data_shape[i]));
	}
	auto py_index_order = PyTuple_New(tdd_dim_data);
	for (int i = 0; i < tdd_dim_data; i++) {
		PyTuple_SetItem(py_index_order, i, PyLong_FromLong(tdd_p_index_order[i]));
	}
	int64_t py_node_code = (int64_t)tdd_node;
	return Py_BuildValue("{sOsLsisOsisOsOsi}",
		"weight", py_weight,
		"node", py_node_code,
		"dim parallel", tdd_dim_parallel,
		"parallel shape", py_parallel_shape,
		"dim data", tdd_dim_data,
		"data shape", py_data_shape,
		"index order", py_index_order,
		"size", tdd_size
	);
}


/// <summary>
/// Get the information of a node. Return a dictionary.
/// </summary>
/// <param name="self"></param>
/// <param name="args"></param>
/// <returns></returns>
static PyObject*
get_node_info(PyObject* self, PyObject* args) {
	//convert from long long
	int64_t code;
	if (!PyArg_ParseTuple(args, "L", &code)) {
		return NULL;
	}
	const Node<W>* p_node = (const Node<W>*)code;

	auto node_id = p_node->get_id();
	auto node_order = p_node->get_order();
	auto node_range = p_node->get_range();
	auto node_p_weights = p_node->get_weights();
	auto node_p_successors = p_node->get_successors();

	auto py_weights = PyTuple_New(node_range);
	auto py_successors = PyTuple_New(node_range);
	for (int i = 0; i < node_range; i++) {
		PyTuple_SetItem(py_weights, i, THPVariable_Wrap(CUDAcpl::from_complex(node_p_weights[i])));
		PyTuple_SetItem(py_successors, i, PyLong_FromLongLong((int64_t)node_p_successors[i]));
	}

	return Py_BuildValue({ "{sisisisOsO}" },
		"id", node_id,
		"order", node_order,
		"range", node_range,
		"out weights", py_weights,
		"successors", py_successors
	);
}




static PyMethodDef ctdd_methods[] = {
	{ "as_tensor", (PyCFunction)as_tensor, METH_VARARGS, "Take in the CUDAcpl tensor, transform to TDD and returns the pointer." },
	{ "to_CUDAcpl", (PyCFunction)to_CUDAcpl, METH_VARARGS, "Return the python torch tensor of the given tdd." },
	{ "tensordot_num", (PyCFunction)tensordot_num, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be a number." },
	{ "tensordot_ls", (PyCFunction)tensordot_ls, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be two index lists." },
	{ "get_tdd_info", (PyCFunction)get_tdd_info, METH_VARARGS, "Get the information of a tdd. Return a dictionary." },
	{ "get_node_info", (PyCFunction)get_node_info, METH_VARARGS, "Get the information of a node. Return a dictionary." },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef superfastcode_module = {
	PyModuleDef_HEAD_INIT,
	"ctdd",                        // Module name to use with Python import statements
	"The C++ backend of tdd.",  // Module description
	0,
	ctdd_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_ctdd() {
	node::Node::reset();
	return PyModule_Create(&superfastcode_module);
}