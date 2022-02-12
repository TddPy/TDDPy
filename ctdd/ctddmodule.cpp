#include "stdafx.h"
#include "tdd.h"

using namespace std;
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
    PyObject* p_tensor, *p_index_order_ls;
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



static PyMethodDef ctdd_methods[] = {
    { "as_tensor", (PyCFunction)as_tensor, METH_VARARGS, "Take in the CUDAcpl tensor, transform to TDD and returns the pointer." },
    { "to_CUDAcpl", (PyCFunction)to_CUDAcpl, METH_VARARGS, "Return the python torch tensor of the given tdd." },
    { "tensordot_num", (PyCFunction)tensordot_num, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be a number." },
    { "tensordot_ls", (PyCFunction)tensordot_ls, METH_VARARGS, "Return the tensordot of two tdds. The index indication should be two index lists." },
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