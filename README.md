# cyper

**Cy**thon **per**formed inline: compile your Cython snippets on the fly.

It is originally inspired by the IPython Cython magic command.


## Usage

### Basic usage

Basic usage:
```python
    code = r'''
    def func(x):
        return 2.0 * x
    '''
    pyx = cyper.inline(code)
    pyx.func(1)
```
Raw string is recommended to avoid breaking escape character.

Export the names from compiled module:
```python
    cyper.inline(code, globals())
    func(1)
```

Get better performance (at your own risk) with arrays:
```python
    cyper.inline(code, fast_indexing=True)
```

Example of using external gsl library, assuming gsl is installed at `/opt/gsl/`
```python
    code = r'''
    cdef extern from "gsl/gsl_math.h":
        double gsl_pow_int (double x, int n)

    def pow(double x, int n):
        y = gsl_pow_int(x, n)
        return y
    '''
    pyx = cyper.inline(
        code,
        include_dirs=['/opt/gsl/include/'],
        library_dirs=['/opt/gsl/lib'],
        libraries=['gsl', 'gslcblas']
    )
    pyx.pow(2, 6)
```

### Advanced usage

Compile OpenMP codes with gcc:
```python
    cyper.inline(openmpcode, openmp='-fopenmp')
    # or equivalently
    cyper.inline(openmpcode,
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                )
    # use '-openmp' or '-qopenmp' (>=15.0) for Intel
    # use '/openmp' for Microsoft Visual C++ Compiler
    # use '-fopenmp=libomp' for Clang
```

The cython `directives` and distutils `extension_args` can also be
set in a directive comment at the top of the code snippet, e.g.:
```python
    # cython: boundscheck=False, wraparound=False, cdivision=True
    # distutils: extra_compile_args = -fopenmp
    # distutils: extra_link_args = -fopenmp
    ...code...
```

Use icc to compile:
```python
    cyper.inline(code, environ={'CC':'icc', 'LDSHARED':'icc -shared'})
```
See https://software.intel.com/en-us/articles/thread-parallelism-in-cython

Set directory for searching cimport (.pxd file):
```python
    cyper.inline(code, cimport_dirs=[custom_path]})
    # or equivalently
    cyper.inline(code, cythonize_args={'include_path': [custom_path]})
```
Try setting `cimport_dirs=sys.path` if Cython can not find installed
cimport module.

## Installation

Dependencies: Cython
