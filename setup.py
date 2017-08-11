from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("nfwfitter.nfwmodeltools", ["nfwfitter/nfwmodeltools.pyx"],
              include_dirs = [numpy.get_include()]
              ),
    Extension("nfwfitter.stats", ["nfwfitter/stats.pyx"],
              include_dirs = [numpy.get_include()]
              ),
    Extension("nfwfitter.voigt_tools",
              ["nfwfitter/voigt_tools.pyx", "nfwfitter/voigt.c"],
              include_dirs = [numpy.get_include()]
              ),
    Extension("nfwfitter.deconvolvedlognormtools", ["nfwfitter/deconvolvedlognormtools.pyx"],
              include_dirs = [numpy.get_include()]
              ),    
    ]

setup(
    name = 'nfwfitter',
    version = '0.1',
    packages = find_packages(),
    ext_modules = cythonize(extensions),
    test_suite = 'nose.collector',
    tests_require = ['nose']
)

