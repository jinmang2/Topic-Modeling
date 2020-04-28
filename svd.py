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

"""
Singular Value Decomposition
"""

full_matrix = True
compute_uv = True
hermitian = False # a가 에르미트 행렬인지 아닌지
                  # A = A^\star
                  # 실수 대칭 행렬의 일반화
                  # 복소수 정사각 행렬

a = np.array(
    [[1, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0],
     [0, 1, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0, 1],
     [0, 0, 0, 1, 1, 0, 1],
     [1, 0, 1, 0, 1, 1, 1]]
)

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

a, wrap = _makearray(a)

if hermitian:
    # 지금 a는 에르미트 행렬이 아님.
    # 아래로 넘어감.
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
# print(t, result_t)

extobj = get_linalg_error_extobj(_raise_linalgerror_svd_nonconvergence)

m, n = a.shape[-2:]
# print(m, n)

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
    # print(wrap(u), s, wrap(vh))
else:
    # FUNC_ARRAY_NAME(svd_N)
    if m < n:
        gufunc = _umath_linalg.svd_m
    else:
        gufunc = _umath_linalg.svd_n
    
    signature = 'D->d' if ixComplexType(t) else 'd->d'
    s = gufunc(a, signature=signature, extobj=extobj)
    s = s.astype(_realType(result_t), copy=False)
    # print(s)

"""
C에서 어떻게 계산하는가?
numpy/linalg/umath_linalg.c.src

```c
/**begin repeat
   #TYPE = FLOAT, DOUBLE, CFLOAT, CDOUBLE#
   #REALTYPE = FLOAT, DOUBLE, FLOAT, DOUBLE#
   #lapack_func = sgesdd, dgesdd, cgesdd, zgesdd#
 */
static NPY_INLINE void
release_@lapack_func@(GESDD_PARAMS_t* params)
{
    /* A and WORK contain allocated blocks */
    free(params->A);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}

static NPY_INLINE void
@TYPE@_svd_wrapper(char JOBZ,
                   char **args,
                   npy_intp* dimensions,
                   npy_intp* steps)
{
    ptrdiff_t outer_steps[4];
    int error_occurred = get_fp_invalid_and_clear();
    size_t iter;
    size_t outer_dim = *dimensions++;
    size_t op_count = (JOBZ=='N')?2:4;
    GESDD_PARAMS_t params;

    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    steps += op_count;

    if (init_@lapack_func@(&params,
                           JOBZ,
                           (fortran_int)dimensions[0],
                           (fortran_int)dimensions[1])) {
        LINEARIZE_DATA_t a_in, u_out, s_out, v_out;
        fortran_int min_m_n = params.M < params.N ? params.M : params.N;

        init_linearize_data(&a_in, params.N, params.M, steps[1], steps[0]);
        if ('N' == params.JOBZ) {
            /* only the singular values are wanted */
            init_linearize_data(&s_out, 1, min_m_n, 0, steps[2]);
        } else {
            fortran_int u_columns, v_rows;
            if ('S' == params.JOBZ) {
                u_columns = min_m_n;
                v_rows = min_m_n;
            } else { /* JOBZ == 'A' */
                u_columns = params.M;
                v_rows = params.N;
            }
            init_linearize_data(&u_out,
                                u_columns, params.M,
                                steps[3], steps[2]);
            init_linearize_data(&s_out,
                                1, min_m_n,
                                0, steps[4]);
            init_linearize_data(&v_out,
                                params.N, v_rows,
                                steps[6], steps[5]);
        }

        for (iter = 0; iter < outer_dim; ++iter) {
            int not_ok;
            /* copy the matrix in */
            linearize_@TYPE@_matrix(params.A, args[0], &a_in);
            not_ok = call_@lapack_func@(&params);
            if (!not_ok) {
                if ('N' == params.JOBZ) {
                    delinearize_@REALTYPE@_matrix(args[1], params.S, &s_out);
                } else {
                    if ('A' == params.JOBZ && min_m_n == 0) {
                        /* Lapack has betrayed us and left these uninitialized,
                         * so produce an identity matrix for whichever of u
                         * and v is not empty.
                         */
                        identity_@TYPE@_matrix(params.U, params.M);
                        identity_@TYPE@_matrix(params.VT, params.N);
                    }

                    delinearize_@TYPE@_matrix(args[1], params.U, &u_out);
                    delinearize_@REALTYPE@_matrix(args[2], params.S, &s_out);
                    delinearize_@TYPE@_matrix(args[3], params.VT, &v_out);
                }
            } else {
                error_occurred = 1;
                if ('N' == params.JOBZ) {
                    nan_@REALTYPE@_matrix(args[1], &s_out);
                } else {
                    nan_@TYPE@_matrix(args[1], &u_out);
                    nan_@REALTYPE@_matrix(args[2], &s_out);
                    nan_@TYPE@_matrix(args[3], &v_out);
                }
            }
            update_pointers((npy_uint8**)args, outer_steps, op_count);
        }

        release_@lapack_func@(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}
/**end repeat*/


/* svd gufunc entry points */
/**begin repeat
   #TYPE = FLOAT, DOUBLE, CFLOAT, CDOUBLE#
 */
static void
@TYPE@_svd_N(char **args,
             npy_intp *dimensions,
             npy_intp *steps,
             void *NPY_UNUSED(func))
{
    @TYPE@_svd_wrapper('N', args, dimensions, steps);
}

static void
@TYPE@_svd_S(char **args,
             npy_intp *dimensions,
             npy_intp *steps,
             void *NPY_UNUSED(func))
{
    @TYPE@_svd_wrapper('S', args, dimensions, steps);
}

static void
@TYPE@_svd_A(char **args,
             npy_intp *dimensions,
             npy_intp *steps,
             void *NPY_UNUSED(func))
{
    @TYPE@_svd_wrapper('A', args, dimensions, steps);
}

/**end repeat**/
```
"""

np.set_printoptions(precision=2)
U, Sigma, V = wrap(u), s, wrap(vh)

def svd_diag(sigma, m, n):
    sigma = np.diag(sigma)
    if m < n:
        sigma = np.c_[sigma, np.zeros((m, 1))]
    elif m > n:
        sigma = np.r_[sigma, np.zeros((1, n))]
    return sigma

Sigma = svd_diag(Sigma, m, n)

ep = np.finfo(float).eps

def is_equal_array_over_eps(a, b, ep=1e-8):
    # Relative Comparision
    diff = abs(a - b)
    maximal = np.max(np.c_[abs(a), abs(b)])
    rel_error = diff / maximal
    return np.all(rel_error < ep)

print(
    is_equal_array_over_eps(a, U.dot(Sigma).dot(V))
)