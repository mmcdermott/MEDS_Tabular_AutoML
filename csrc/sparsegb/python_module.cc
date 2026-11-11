// Lets setuptools build and install sparsegb as an extension module. Python code never imports it:
// MEDS_tabular_automl/sparsegb/__init__.py loads this shared library with ctypes and calls its C
// ABI.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef sparsegb_module = {
    PyModuleDef_HEAD_INIT,
    "_sparsegb",
    "sparsegb gradient boosting library; loaded with ctypes by MEDS_tabular_automl.sparsegb.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

PyMODINIT_FUNC PyInit__sparsegb(void) { return PyModule_Create(&sparsegb_module); }
