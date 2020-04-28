import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    """
    The one-dimensional SVD 
    
    Theorem: 
        Let x be a random unit vector and let $B={A}^T{A}={V}{\Sigma}^2{V}^T$.
        Then with high probability, $\lim_{s\to\infty}{B}^s{x}$ is in the
            span of the first singular vector $v_1$.
        If we normalize ${B}^s{x}$ to a unit vector at each $s$,
            then furthermore the limit is $v_1$.
    """

    # Get random unit vector x s.t x = \sum_i {c_i}{v_i}
    x = randomUnitVector(min(n, m)) # sum(quadratic) is 1.
    n, m = A.shape
    currentV = x

    # Calc B = {A}^T{A}
    if n > m:
        B = np.dot(matrixFor1D.T, matrixFor1D) # (n, m)^T x (n, m)
                                               # (m, n)   x (n, m)
                                               # (m, m) lower-dimension
    else:
        B = np.dot(matrixFor1D, matrixFor1D.T) # (n, m) x (n, m)^T
                                               # (n, m) x (m, n)
                                               # (n, n) lower-dimension

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        # Calc {B}^s{x} = \sum_i {c_i}{\sigma_i}^{2s}{v_i}
        # If you multiply B recursively, since U (or V) is orthogonal square matrix,
        # {U^T}{U} (or {V^T}{V}) = I. Hence above B equation generated.
        currentV = np.dot(B, lastV)
        # Normalize
        currentV = currentV / norm(currentV)
        # Calc {(\sigma_{j})/\sigma_{j+1}}^{2s}. is larger than 1(-ep)?
        # Then converge.
        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print(f"converged in {iterations} iterations!")
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        # If K is None, computes the full-rank decomposition.
        # If k is less than min(n, m), perform truncated-SVD.
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            # {A^\prime} = {A} - {\sigma_1(A)}{u_1}{v_1}^T
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs




