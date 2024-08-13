# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name>
<Class>
<Date>
"""
import numpy as np
import scipy as sp
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    A = A.astype(float)
    m,n = A.shape
    Q = A.copy()
    R = np.zeros((n,n))
    for i in range(n):
        R[i][i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i][i]
        for j in range(i+1, n):
            R[i][j] = Q[:,j].T @ Q[:, i]
            Q[:,j] = Q[:,j] - R[i][j]*Q[:,i]
    return Q, R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    return np.prod(np.array([abs(qr_gram_schmidt(A)[1][i][i]) for i in range(len(A[0]))]))

# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q,R = qr_gram_schmidt(A)
    y = Q.T@b
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(1,n+1):
        for j in range(1, i):
            x[n-i] += R[n-i][n-j]*x[n-j]
        x[n-i] = (y[n-i] - x[n-i])/(R[n-i][n-i])
    return x

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    sign = lambda x: 1 if x >= 0 else -1
    m,n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        u = R[k:,k].copy()
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*u.reshape((len(u),1))@(u.T@R[k:,k:]).reshape((1,m-k))
        Q[k:,:] = Q[k:,:] - 2*u.reshape((len(u),1))@(u.T@Q[k:,:]).reshape((1,m))
    return Q.T, R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x: 1 if x >= 0 else -1
    m,n = A.shape
    H = A.copy()
    Q = np.eye(m)
    for k in range(n-2):
        u = H[k+1:,k].copy()
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*u.reshape((len(u),1))@(u.T@H[k+1:,k:]).reshape((1,m-k))
        H[:,k+1:] = H[:,k+1:] - 2*(H[:,k+1:]@u).reshape((m,1))@u.reshape((1,len(u)))
        Q[k+1:,:] = Q[k+1:,:] - 2*u.reshape((len(u),1))@(u.T@Q[k+1:,:]).reshape(1, m)
    return H, Q.T





