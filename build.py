from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
import numpy
import os


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """

    extensions = ["./hopsflow/gassflow_two_formulas.pyx"]

    # gcc arguments hack: enable optimizations
    os.environ["CFLAGS"] = f"-O3 -I{numpy.get_include()}"

    # Build
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                extensions,
                language_level=3,
                compiler_directives={"linetrace": True},
            ),
            "cmdclass": {"build_ext": build_ext},
        }
    )
