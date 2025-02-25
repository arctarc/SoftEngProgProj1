from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import os
os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
os.environ["CXX"] = "/opt/homebrew/bin/g++-14"

# Telling the compiler to use GCC instead of Clang, since Clang doesn't support OpenMP

ext_modules = [
    Extension(
        "OptMPI",
        ["OptMPI.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

# This extra section, along with an extra import, is required for enabling OpenMP

setup(
    name="OptMPI",
    ext_modules=cythonize(ext_modules, language_level="3"),
    include_dirs=[numpy.get_include()],
)
include_dirs=[numpy.get_include()],
)