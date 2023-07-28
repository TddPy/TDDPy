// The interfaces of the ctdd kernel.

#include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/pybind11.h>            // Pybind11 import to define Python bindings
#include <xtensor/xmath.hpp>              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>     // Numpy bindings

double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

PYBIND11_MODULE(ctdd_dll, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
}