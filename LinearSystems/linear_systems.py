# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""

import numpy as np
import scipy as sp
from scipy import linalg as la

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    A = np.array(A)
    A = A.astype(float)
    for j in range(len(A)):
        for i in range(j+1, len(A)):
            A[i] -= A[j] * (A[i][j]/A[j][j])

    return A

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    A = np.array(A)
    A = A.astype(float)
    (n,n) = np.shape(A)
    U = np.copy(A)
    L = np.eye(n, dtype=float)
    for j in range(n):
        for i in range(j+1, n):
            L[i][j] = U[i][j]/U[j][j]
            U[i][j:] = U[i][j:] - L[i][j]*U[j][j:]

    return L, U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L, U = lu(A)
    y = []
    for k in range(len(b)):
        lysum = 0
        for j in range(k):
            lysum += L[k][j] * y[j]
        y.append(b[k] - lysum)

    x = []
    for k in range(len(y)-1, -1, -1):
        uxsum = 0
        for j in range(k+1, len(y)):
            uxsum += U[k][j]*x[len(y)-1 - j]
        x.append((y[k] - uxsum)/U[k][k])
    x = np.array(x[::-1])
    return x.flatten()

# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    import time
    from matplotlib import pyplot as plt
    t_inv = []
    t_solve = []
    t_luf = []
    t_lus = []
    X = [2**i for i in range(12)]
    for n in X:
        A = np.random.random((n,n))
        b = np.random.random(n)
        t1 = time.time()
        A_inv = la.inv(A)
        A_inv @ b
        t2 = time.time()
        la.solve(A, b)
        t3 = time.time()
        L,P = la.lu_factor(A)
        t4 = time.time()
        la.lu_solve((L,P), b)
        t5 = time.time()

        t_inv.append(t2-t1)
        t_solve.append(t3-t2)
        t_luf.append(t5-t3)
        t_lus.append(t5-t4)

    plt.ion()
    ax1 = plt.subplot(121)
    ax1.plot(X, t_inv, label='inverse')
    ax1.plot(X, t_solve, label='solve')
    ax1.plot(X, t_luf, label='LU factorization')
    ax1.plot(X, t_lus, label='LU solve')
    ax1.legend(loc='upper left')

    ax2 = plt.subplot(122)
    ax2.loglog(X, t_inv, basex=2, basey=2)
    ax2.loglog(X, t_solve, basex=2, basey=2)
    ax2.loglog(X, t_luf, basex=2, basey=2)
    ax2.loglog(X, t_lus, basex=2, basey=2)

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    from scipy import sparse
    Iden = sparse.diags([1], [0], shape=(n,n))
    B = sparse.diags([1,-4,1], [-1,0,1], shape=(n,n))
    A = sparse.lil_matrix((n**2, n**2))
    for i in range(n):
        A[i*n:(i+1)*n,i*n:(i+1)*n] = B
        if i > 0:
            A[(i-1)*n:i*n,i*n:(i+1)*n] = Iden
            A[i*n:(i+1)*n,(i-1)*n:i*n] = Iden
    return A

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    from matplotlib import pyplot as plt
    from scipy.sparse import linalg as spla
    import time

    t_csr = []
    t_npa = []
    n = [2**i for i in range(1,7)]
    for i in n:
        A = prob5(i)
        b = np.random.random(i**2)
        t1 = time.time()
        Acsr = A.tocsr()
        spla.spsolve(Acsr, b)
        t2 = time.time()
        Anpa = A.toarray()
        la.solve(Anpa, b)
        t3 = time.time()
        t_csr.append(t2-t1)
        t_npa.append(t3-t2)

    plt.ion()
    plt.loglog(n, t_csr, basex=2, basey=2, label='CSR')
    plt.loglog(n, t_npa, basex=2, basey=2, label='NP array')
    plt.legend(loc="upper left")

