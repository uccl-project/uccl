/* Minimal stable-ABI stub so setuptools emits a cp38-abi3 platform tag. */
#define Py_LIMITED_API 0x03080000
#include <Python.h>

static PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "uccl._abi3_stub",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__abi3_stub(void) {
    return PyModule_Create(&_module);
}
