#include "stdafx.h"
#include "tdd.h"

using namespace std;
using namespace tdd;

static PyObject*
test(PyObject* self, PyObject* args)
{
    cout << 1 << endl;
    PyObject* p_temp;
    if (!PyArg_ParseTuple(args, "O", &p_temp))
        return NULL;
    cout << 2 << endl;
    auto t = THPVariable_Unpack(p_temp);
    t[0] = 0.;
    cout << t << endl;
    return PyLong_FromLong(10);
}

static PyObject*
as_tensor(PyObject* self, PyObject* args)
{
    PyObject* p_tensor, *p_index_order_ls;
    int dim_parallel;
    if (!PyArg_ParseTuple(args, "OiO", &p_tensor, &dim_parallel, &p_index_order_ls))
        return NULL;
    auto t = THPVariable_Unpack(p_tensor);
    //prepare the index_order array
    auto size = PyList_GET_SIZE(p_index_order_ls);
    int* p_index_order = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) {
        p_index_order[i] = _PyLong_AsInt(PyList_GetItem(p_index_order_ls, i));
    }
    //construct the tdd
    TDD* p_res = TDD::as_tensor(t, dim_parallel, p_index_order);
    cout << "OOO" << p_res->get_size() << endl;
    free(p_index_order);

    return Py_BuildValue("O",p_res);
}




static PyMethodDef ctdd_methods[] = {
    { "test", (PyCFunction)test, METH_VARARGS, "test the function." },
    { "as_tensor", (PyCFunction)as_tensor, METH_VARARGS, "Take in the CUDAcpl tensor, transform to TDD and returns the pointer." },

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