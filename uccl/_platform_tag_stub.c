/*
 * Minimal stub so setuptools emits a platform-specific wheel tag.
 * On Python >= 3.12 this is built with -DPy_LIMITED_API=0x030C0000
 * (stable ABI, cp312-abi3 tag); on older Pythons it compiles without
 * the limited-API flag (version-specific cpXY-cpXY tag).
 */

#include <Python.h>

static PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "uccl._platform_tag_stub",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__platform_tag_stub(void) { return PyModule_Create(&_module); }
