"""
cyper.inline can easily compile your cython snippets on the fly,
without writing the tedious setup files or makefile.
It is a standalone package originally inspired by IPython Cython magic.
"""
from __future__ import absolute_import, print_function

import re
import io
import os
import sys
import hashlib
import inspect
import contextlib
from distutils.core import Extension

import Cython
from Cython.Utils import captured_fd, get_cython_cache_dir
from Cython.Build import cythonize
from Cython.Build.Inline import to_unicode, strip_common_indent
from Cython.Build.Inline import _get_build_extension


__all__ = ['inline']


def _append_args(kwargs, key, value):
    kwargs[key] = [value] + kwargs.get(key, [])


def _extend_args(kwargs, key, value_list):
    kwargs[key] = value_list + kwargs.get(key, [])


def _export_all(source, target):
    """Import all variables from the namespace `source` to `target`.
    Both arguments must be dict-like objects.
    If `source['__all__']` is defined, only variables in it will be imported, otherwise
    all variables not starting with '_' will be imported.
    """
    if '__all__' in source:
        keys = source['__all__']
    else:
        keys = [k for k in source if not k.startswith('_')]

    for k in keys:
        try:
            target[k] = source[k]
        except KeyError:
            msg = "'module' object has no attribute '%s'" % k
            raise AttributeError(msg)


def join_path(path1, path2):
    """Join and normalize two paths.
    """
    return os.path.normpath(os.path.join(
        path1, os.path.expanduser(path2)))


def get_basename(path):
    """Get the base name of the file, e.g. 'abc' for 'dir/abc.py'.
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_frame_dir(depth=0):
    """Return the source file directory of a frame in the call stack.
    """
    if hasattr(sys, "_getframe"):
        frame = sys._getframe(depth + 1)  # +1 for this function itself
    else:
        raise NotImplementedError("Support CPython only.")
    file = inspect.getabsfile(frame)
    return os.path.dirname(file)


def so_ext():
    """Get extension for the compiled library.
    """
    if not hasattr(so_ext, 'ext'):
        so_ext.ext = _get_build_extension().get_ext_filename('')
    return so_ext.ext


def load_dynamic(name, path):
    """Load and initialize a module implemented as a dynamically loadable
    shared library and return its module object. If the module was already
    initialized, it will be initialized again.
    """
    # imp module is deprecated since Python 3.4
    if (sys.version_info >= (3, 4)):
        from importlib.machinery import ExtensionFileLoader
        from importlib.util import spec_from_loader, module_from_spec
        loader = ExtensionFileLoader(name, path)
        spec = spec_from_loader(name, loader, origin=path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        import imp
        return imp.load_dynamic(name, path)


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the environment variables.
    source: http://stackoverflow.com/a/34333710/

    Examples
    --------
    >>> with set_env(PLUGINS_DIR=u'plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True
    >>> "PLUGINS_DIR" in os.environ
    False
    """
    try:
        if environ:
            old_environ = dict(os.environ)
            os.environ.update(environ)
        yield
    finally:
        if environ:
            os.environ.clear()
            os.environ.update(old_environ)


@contextlib.contextmanager
def _suppress_output(quiet=True):
    """Suppress any output/error/warning in compiling
    if quiet is True and no exception raised.
    """
    try:
        # `captured_fd` only captures the default IO streams, we must redirect
        # the streams to defaults for jupyter notebook to enable capturing.
        old_stream = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

        get_outs = get_errs = lambda: None  # backup for failure
        with captured_fd(1) as get_outs:
            with captured_fd(2) as get_errs:
                yield

    except Exception:
        quiet = False
        raise

    finally:
        sys.stdout, sys.stderr = old_stream

        if not quiet:
            outs, errs = get_outs(), get_errs()
            if outs:
                print("Compiler Output\n===============",
                      outs.decode('utf8'), sep='\n', file=sys.stdout)
            if errs:
                print("Compiler Error/Warning\n======================",
                      errs.decode('utf8'), sep='\n', file=sys.stderr)


def _update_flag(code, args, auto_flag=True):
    """Update compiler options for numpy and openmp.
    Helper function for inline.
    """
    numpy = args.pop('numpy', None)
    openmp = args.pop('openmp', None)

    if numpy is None and auto_flag:
        reg_numpy = re.compile(r"""
            ^\s* cimport \s+ numpy |
            ^\s* from \s+ numpy \s+ cimport
            """, re.M | re.X)
        numpy = reg_numpy.search(code)

    if openmp is None and auto_flag:
        reg_openmp = re.compile(r"""
            ^\s* c?import \s+cython\.parallel |
            ^\s* from \s+ cython\.parallel \s+ c?import |
            ^\s* from \s+ cython \s+ c?import \s+ parallel
            """, re.M | re.X)
        openmp = reg_openmp.search(code)

    if numpy:
        import numpy
        _append_args(args, 'include_dirs', numpy.get_include())

    if openmp:
        if hasattr(openmp, 'startswith'):
            openmp_flag = openmp  # openmp is string
        else:
            openmp_flag = '-fopenmp'
        _append_args(args, 'extra_compile_args', openmp_flag)
        _append_args(args, 'extra_link_args', openmp_flag)


def cython_build(name, file=None, force=False, quiet=True, cythonize_args={},
                 lib_dir=os.path.join(get_cython_cache_dir(), 'inline/lib'),
                 tmp_dir=os.path.join(get_cython_cache_dir(), 'inline/tmp'),
                 **extension_args):
    """Build a cython extension.
    """
    if file is not None:
        _append_args(extension_args, 'sources', file)

    with _suppress_output(quiet=quiet):
        extension = Extension(name, **extension_args)
        extensions = cythonize([extension], force=force, **cythonize_args)

        build_extension = _get_build_extension()
        build_extension.extensions = extensions
        build_extension.build_lib = lib_dir
        build_extension.build_temp = tmp_dir
        build_extension.run()

        # ext_file = os.path.join(lib_dir, name + so_ext())
        # module = load_dynamic(name, ext_file)
        # return module


def inline(code, export=None, name=None, force=False,
           quiet=True, auto_flag=True, fast_indexing=False,
           directives={}, cimport_dirs=[], cythonize_args={},
           lib_dir=os.path.join(get_cython_cache_dir(), 'inline/lib'),
           tmp_dir=os.path.join(get_cython_cache_dir(), 'inline/tmp'),
           environ={}, **extension_args):
    """Compile a code snippet in string.
    The contents of the code are written to a `.pyx` file in the
    cython cache directory using a filename with the hash of the
    code. This file is then cythonized and compiled.

    Parameters
    ----------
    code : str
        The code to compile.
        It can also be a file path, but must start with "./", "/", "X:", or "~",
        and end with ".py" or ".pyx".
        Strings like "import abc.pyx" or "a=1; b=a.pyx" will be treated as
        code snippet.
    export : dict
        Export the variables from the compiled module to a dict.
        `export=globals()` is equivalent to `from module import *`.
    name : str, optional
        Name of compiled module. If not given, it will be generated
        automatically by hash of the code and options (recommended).
    force : bool
        Force the compilation of a new module, even if the source
        has been previously compiled.
    quiet : bool
        Suppress compiler's outputs/warnings unless the compiling failed.
    auto_flag : bool
        If True, numpy and openmp will be auto-detected from the code.
    fast_indexing : bool
        If True, `boundscheck` and `wraparound` are turned off
        for better array indexing performance (at cost of safety).
        This setting can be overridden by `directives`.
    directives : dict
        Cython compiler directives, including
            binding, boundscheck, wraparound, initializedcheck, nonecheck,
            overflowcheck, overflowcheck.fold, embedsignature, cdivision, cdivision_warnings,
            always_allow_keywords, profile, linetrace, infer_types, language_level, etc.
        Ref http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
        This setting can be overridden by `cythonize_args['compiler_directives']`.
    cimport_dirs : list of str
        Directories for finding cimport modules (.pxd files).
        This setting can be overridden by `cythonize_args['include_path']`.
    cythonize_args : dict
        Arguments for `Cython.Build.cythonize`, including
            aliases, quiet, force, language, annotate, build_dir, output_file,
            include_path, compiler_directives, etc.
        Ref http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
    environ : dict
        Temporary environment variables for compilation.
    lib_dir : str
        Directory to put the compiled module.
    tmp_dir : str
        Directory to put the temporary files.
    **extension_args :
        Arguments for `distutils.core.Extension`, including
            name, sources, define_macros, undef_macros,
            include_dirs, library_dirs, runtime_library_dirs,
            libraries, extra_compile_args, extra_link_args,
            extra_objects, export_symbols, depends, language
        Ref https://docs.python.org/2/distutils/apiref.html#distutils.core.Extension

    Examples
    --------
    Basic usage:
        code = r'''
        def func(x):
            return 2.0 * x
        '''
        pyx = cyper.inline(code)
        pyx.func(1)
    Raw string is recommended to avoid breaking escape character.

    Export the names from compiled module:
        cyper.inline(code, globals())
        func(1)

    Get better performance (at your own risk) with arrays:
        cyper.inline(code, fast_indexing=True)

    Example of using gsl library, assuming gsl is installed at /opt/gsl/
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

    Compile OpenMP codes with gcc:
        cyper.inline(openmpcode, openmp='-fopenmp')
        # or equivalently
        cyper.inline(openmpcode,
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    )
        # use '-openmp' or '-qopenmp' (>=15.0) for Intel
        # use '/openmp' for Microsoft Visual C++ Compiler
        # use '-fopenmp=libomp' for Clang

    The cython `directives` and distutils `extension_args` can also be
    set in a directive comment at the top of the code snippet, e.g.:
        # cython: boundscheck=False, wraparound=False, cdivision=True
        # distutils: extra_compile_args = -fopenmp
        # distutils: extra_link_args = -fopenmp
        ...code...

    Use icc to compile:
        cyper.inline(code, environ={'CC':'icc', 'LDSHARED':'icc -shared'})
    See https://software.intel.com/en-us/articles/thread-parallelism-in-cython

    Set directory for searching cimport (.pxd file):
        cyper.inline(code, cimport_dirs=[custom_path]})
        # or equivalently
        cyper.inline(code, cythonize_args={'include_path': [custom_path]})
    Try setting `cimport_dirs=sys.path` if Cython can not find installed
    cimport module.

    See also
    --------
    https://github.com/cython/cython/blob/master/Cython/Build/IpythonMagic.py
    https://github.com/cython/cython/blob/master/Cython/Build/Inline.py
    """
    # get working directories
    # assume all paths are relative to the directory of the caller's frame
    cur_dir = get_frame_dir(depth=1)  # where inline is called

    lib_dir = join_path(cur_dir, lib_dir)
    tmp_dir = join_path(cur_dir, tmp_dir)

    if not os.path.isdir(lib_dir):
        os.makedirs(lib_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    # check if `code` is a code snippet or a .pyx/.py file
    reg_pyx = re.compile(r"^ ( ~ | [\.]? [/\\] | [a-zA-Z]:) .* \.pyx? $ | "
                         r"^ [^\s=;]+ \.pyx? $", re.X | re.S)
    is_file = reg_pyx.match(code)

    if is_file:
        file = join_path(cur_dir, code)
        code = io.open(file, 'r', encoding='utf-8').read()
        if name is None:
            name = get_basename(file)
        # it might exist related .pyd file in the same directory
        cimport_dirs = cimport_dirs + [os.path.dirname(file)]
    else:
        cimport_dirs = cimport_dirs + [cur_dir]
    code = strip_common_indent(to_unicode(code))

    # update arguments
    directives = directives.copy()
    if fast_indexing:
        directives.setdefault('boundscheck', False)
        directives.setdefault('wraparound', False)
    directives.setdefault('embedsignature', True)  # recommended setting

    cythonize_args = cythonize_args.copy()
    cythonize_args.setdefault('compiler_directives', directives)
    cythonize_args.setdefault('include_path', cimport_dirs)

    # if any extra dependencies
    extra_depends = any(extension_args.get(k, [])
                        for k in ['sources', 'extra_objects', 'depends'])

    # module signature
    key = (code, name, cythonize_args, extension_args, environ, os.environ,
           sys.executable, sys.version_info, Cython.__version__)
    key_bytes = u"{}".format(key).encode('utf-8')   # for 2, 3 compatibility
    signature = hashlib.md5(key_bytes).hexdigest()

    # embed module signature?
    # code = u"{}\n\n# added by cyper.inline\n{} = '{}'".format(
    #     code, '__cyper_signature__', signature)

    # module name and path
    pyx_name = "_cyper_{}".format(signature)
    ext_name = pyx_name if name is None else name

    pyx_file = os.path.join(tmp_dir, pyx_name + '.pyx')  # path of source file
    ext_file = os.path.join(lib_dir, ext_name + so_ext())  # path of extension

    # write pyx file
    if force or not os.path.isfile(pyx_file):
        with io.open(pyx_file, 'w', encoding='utf-8') as f:
            f.write(code)
        if os.path.isfile(ext_file):
            os.remove(ext_file)  # dangerous?

    # build
    # if existing extra depends, let distutils to decide whether rebuild or not
    if not os.path.isfile(ext_file) or extra_depends:
        with set_env(**environ):
            _update_flag(code, extension_args, auto_flag=auto_flag)
            cython_build(ext_name, file=pyx_file, force=force,
                         quiet=quiet, cythonize_args=cythonize_args,
                         lib_dir=lib_dir, tmp_dir=tmp_dir,
                         **extension_args)

    # import
    module = load_dynamic(ext_name, ext_file)
    # module.__pyx_file__ = pyx_file
    if export is not None:
        _export_all(module.__dict__, export)
    return module
