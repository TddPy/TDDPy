#include <Python.h>
#include <Windows.h>
#include <cmath>

static PyMethodDef ctdd_methods[] = {
    // The first property is the name exposed to Python, fast_tanh
    // The second is the C++ function with the implementation
    // METH_O means it takes a single PyObject argument
    //{ "fast_tanh", (PyCFunction)tanh_impl, METH_O, nullptr },

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
    return PyModule_Create(&superfastcode_module);
}