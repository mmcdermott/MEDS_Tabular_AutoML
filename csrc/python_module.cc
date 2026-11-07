// Lets setuptools build and install the engine as an extension module. Python code never
// imports it: native/engine.py loads this shared library with ctypes and calls the C ABI
// declared in meds_tab.h.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef engine_module = {
    PyModuleDef_HEAD_INIT,
    "_engine",
    "MEDS-Tab's native engine; loaded with ctypes by MEDS_tabular_automl.native.engine.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

PyMODINIT_FUNC PyInit__engine(void) { return PyModule_Create(&engine_module); }
