<<<<<<< HEAD
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("Cyphon.pyx", compiler_directives={'language_level': "3"}),
)
=======
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("Cyphon.pyx", compiler_directives={'language_level': "3"}),
)
>>>>>>> 124e4a5855a6e6960ac23bb1d3b8a3688fb6d878
