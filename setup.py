"""Build PicoSGL C++ extensions.

Usage (run from project root):
    python setup.py build_ext --inplace

After building, the .so lands in python/picosgl/ and is importable as
``picosgl._cpp_kvcache``.

NOTE: editable install (``pip install -e .``) is NOT supported for C++
extensions with relative include paths — use ``build_ext --inplace`` instead.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name="picosgl.plusplus.kvcache",
        sources=["csrc/src/kvcache/pool.cpp"],
        include_dirs=["csrc/include"],
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="picosgl",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
