# cyper

**Cy**thon **per**formed inline: compile and run your Cython snippets on the fly.

It is aiming to provide a flexible alternative to the IPython Cython magic command outside the IPython/Jupyter environment.

## Features

- Simple, get rid of the tedious setup files or makefile.
- Smart, auto-detect and set compiler flags for Numpy and OpenMP.
- Highly customizable, allow one to set Cython directives, environment variables and compiler options.

## Usage

### Basic usage

- Basic usage
  ```python
  import cyper
  code = r'''
  def func(x):
      return 2.0 * x
  '''
  pyx = cyper.inline(code)
  pyx.func(1)
  # 2.0
  ```
  Raw string is recommended to avoid breaking escape character.

- It is convenient (though usually not encouraged) to export the variables from compiled module to the current namespace
  ```python
  cyper.inline(code, globals())
  func(1)
  ```

- Example of using Numpy array and external gsl library, assuming gsl installed at `/opt/gsl/`
  ```python
  code = r'''
  import numpy as np
  
  cdef extern from "gsl/gsl_math.h":
      double gsl_pow_int (double x, int n)
  
  def pow(double x, int n):
      y = gsl_pow_int(x, n)
      return y
  
  def pow_array(double[:] x, int n):
      cdef:
          int i, m=len(x)
          double[:] y=np.empty(m, dtype='f8')
      for i in range(m):
          y[i] = gsl_pow_int(x[i], n)
      return y.base
  '''
  pyx = cyper.inline(
      code,
      include_dirs=['/opt/gsl/include/'],
      library_dirs=['/opt/gsl/lib'],
      libraries=['gsl', 'gslcblas']
  )
  
  pyx.pow(2, 6)
  # 64.0
  
  import numpy as np
  pyx.pow_array(np.arange(5, dtype='f8'), 2)
  # array([ 0.,  1.,  4.,  9., 16.])
  ```
  
- Get better performance (at your own risk) with arrays
  ```python
  cyper.inline(code, fast_indexing=True)
  # or equivalently
  cyper.inline(code, directives=dict(boundscheck=False, wraparound=False))
  ```

### Advanced usage

- Set the compiler options, e.g., compiling OpenMP codes with gcc
  ```python
  cyper.inline(openmpcode,
               extra_compile_args=['-fopenmp'],
               extra_link_args=['-fopenmp'],
               )
  # use '-openmp' or '-qopenmp' (>=15.0) for Intel
  # use '/openmp' for Microsoft Visual C++ Compiler
  # use '-fopenmp=libomp' for Clang
  ```
  Or equivalently write this for short
  ```python
  cyper.inline(openmpcode, openmp='-fopenmp')
  ```
  
  The cython `directives` and distutils `extension_args` can also be set in a directive comment at the top of the code snippet, e.g.,
  ```python
  code = r"""
  # cython: boundscheck=False, wraparound=False, cdivision=True
  # distutils: extra_compile_args = -fopenmp
  # distutils: extra_link_args = -fopenmp
  ...<code>...
  """
  cyper.inline(code)
  ```

- Set environment variables, e.g., using icc to compile
  ```python
  cyper.inline(code, environ={'CC':'icc', 'LDSHARED':'icc -shared'})
  ```
  See https://software.intel.com/en-us/articles/thread-parallelism-in-cython

- Set directory for searching cimport (.pxd file)
  ```python
  cyper.inline(code, cimport_dirs=[custom_path]})
  # or equivalently
  cyper.inline(code, cythonize_args={'include_path': [custom_path]})
  ```
  Try setting `cimport_dirs=sys.path` if Cython can not find the installed cimport modules.

## Installation

- Dependencies: Cython
