// Lets setuptools build and install medsboost as an extension module. Python code never imports it:
// MEDS_tabular_automl/medsboost/__init__.py loads this shared library with ctypes and calls its C
// ABI.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef medsboost_module = {
    PyModuleDef_HEAD_INIT,
    "_medsboost",
    "medsboost gradient boosting library; loaded with ctypes by MEDS_tabular_automl.medsboost.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

PyMODINIT_FUNC PyInit__medsboost(void) { return PyModule_Create(&medsboost_module); }
