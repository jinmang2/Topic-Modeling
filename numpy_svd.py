# https://github.com/numpy/numpy/blob/v1.17.0/numpy/linalg/linalg.py#L1468-L1650
# line 1468
import numpy as np
from numpy import linalg as LA
import functools
import operator
import warnings

from numpy.core import (
    array, asarray, zeros, empty, empty_like, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, all, Inf, dot,
    add, multiply, sqrt, fastCopyAndTranspose, sum, isfinite,
    finfo, errstate, geterrobj, moveaxis, amin, amax, product, abs,
    atleast_2d, intp, asanyarray, object_, matmul,
    swapaxes, divide, count_nonzero, isnan, sign
)
from numpy.core.multiarray import normalize_axis_index
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import lapack_lite, _umath_linalg


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.linalg')


def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap


def transpose(a):
    return swapaxes(a, -1, -2)


@set_module('numpy.linalg')
class LinAlgError(Exception):
    pass


def _assertRankAtLeast2(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError(
                "$d-dimensional array given. Array must be"
                "at least two-dimensional" % a.ndim
            )


def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, inexact):
            if isComplexType(a.dtype.type):
                is_complex = True
            rt = _realType(a.dtype.type, default=None)
            if rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            rt = double
        if rt is double:
            result_type = double
    if is_complex:
        t = cdouble
        result_type = _complex_types_map[result_type]
    else:
        t = double
    return t, result_type


def _determine_error_states():
    errobj = geterrobj()
    bufsize = errobj[0]

    with errstate(invalid='call', over='ignore',
                  divide='ignore', under='ignore'):
        invalid_call_errmask = geterrobj()[1]

    return [bufsize, invalid_call_errmask, None]


# Dealing with errors in _umath_linalg
_linalg_error_extobj = _determine_error_states()


def _raise_linalgerror_svd_nonconvergence(err, flag):
    raise LinAlgError("SVD did not converge")


def get_linalg_error_extobj(callback):
    extobj = list(_linalg_error_extobj)  # make a copy
    extobj[2] = callback
    return extobj


def isComplexType(t):
    return issubclass(t, complexfloating)


_real_types_map = {single : single,
                   double : double,
                   csingle : single,
                   cdouble : double}


_complex_types_map = {single : csingle,
                      double : cdouble,
                      csingle : csingle,
                      cdouble : cdouble}


def _realType(t, default=double):
    return _real_types_map.get(t, default)


def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)


@array_function_dispatch(_svd_dispatcher)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):


# Singular value decomposition
def _svd_dispatcher(a, full_matrices=None, compute_uv=None, hermitian=None):
    return (a,)


@array_function_dispatch(_svd_dispatcher)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    full_matrix = True # If True,
                    # (m, m) X (min(m, n),) X (n, n)
                    # Else:
                    # (m, min(m, n)) X (min(m, n),) X (min(m, n), n)
    compute_uv = True
    hermitian = False  # a가 에르미트 행렬인지 아닌지
                    # A = A^\star
                    # 실수 대칭 행렬의 일반화
                    # 복소수 정사각 행렬
    """
    a, wrap = _makearray(a)

    if hermitian:
        if compute_uv:
            s, u = eigh(a)
            s = s[..., ::-1]
            u = u[..., ::-1]
            # singular values are unsigned, move the sign into v
            vt = transpose(u * sign(s)[..., None, :]).conjugate()
            s = abs(s)
            print(wrap(u), s, wrap(vt))
        else:
            s = LA.eigvalsh(a)
            s = s[..., ::-1]
            s = abs(s)
            print(s)

    _assertRankAtLeast2(a)
    t, result_t = _commonType(a)

    extobj = get_linalg_error_extobj(_raise_linalgerror_svd_nonconvergence)

    m, n = a.shape[-2:]

    if compute_uv:
        if full_matrix:
            # FUNC_ARRAY_NAME(svd_A)
            if m < n:
                gufunc = _umath_linalg.svd_m_f
            else:
                gufunc = _umath_linalg.svd_n_f
        else:
            # FUNC_ARRAY_NAME(svd_S)
            if m < n:
                gufunc = _umath_linalg.svd_m_s
            else:
                gufunc = _umath_linalg.svd_n_s
        
        signature = 'D->DdD' if isComplexType(t) else 'd->ddd'
        u, s, vh = gufunc(a, signature=signature, extobj=extobj)
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)
        return wrap(u), s, wrap(vh)
    else:
        # FUNC_ARRAY_NAME(svd_N)
        if m < n:
            gufunc = _umath_linalg.svd_m
        else:
            gufunc = _umath_linalg.svd_n
        
        signature = 'D->d' if ixComplexType(t) else 'd->d'
        s = gufunc(a, signature=signature, extobj=extobj)
        s = s.astype(_realType(result_t), copy=False)
        return s