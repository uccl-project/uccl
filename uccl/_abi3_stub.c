/* Minimal stable-ABI stub so setuptools emits a cp312-abi3 platform tag. */
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030C0000
#endif
#include <Python.h>

static PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "uccl._abi3_stub",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__abi3_stub(void) {
    return PyModule_Create(&_module);
}
