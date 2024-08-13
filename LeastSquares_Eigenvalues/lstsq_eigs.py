# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
import scipy as sp
from scipy import linalg as la
from matplotlib import pyplot as plt
import cmath

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode='economic')
    return la.solve_triangular(R, Q.T@b)

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    housing = np.load('housing.npy')
    A = np.column_stack((housing[:,0], np.ones(len(housing))))
    b = housing[:,1]
    x = least_squares(A, b)

    plt.ion()
    plt.plot(housing[:,0], housing[:,1], 'k*', label="Data Points")
    plt.plot(housing[:,0], x[0]*housing[:,0] + x[1], label="Least Squares Fit")
    plt.legend(loc="upper left")


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    housing = np.load('housing.npy')
    A_3 = np.vander(housing[:,0], 4)
    A_6 = np.vander(housing[:,0], 7)
    A_9 = np.vander(housing[:,0], 10)
    A_12 = np.vander(housing[:,0], 13)
    b = housing[:,1]
    x_3 = la.lstsq(A_3,b)[0]
    x_6 = la.lstsq(A_6,b)[0]
    x_9 = la.lstsq(A_9,b)[0]
    x_12 = la.lstsq(A_12,b)[0]
    rd = np.linspace(0, housing[-1][0], num=100)

    plt.ion()
    ax1 = plt.subplot(221)
    ax1.plot(housing[:,0], housing[:,1], 'k*', label="Data Points")
    p_3 = np.poly1d(x_3)
    ax1.plot(rd, p_3(rd), label="Least Squares Fit")
    ax1.set_title("Degree=3")

    ax2 = plt.subplot(222)
    ax2.plot(housing[:,0], housing[:,1], 'k*', label="Data Points")
    p_6 = np.poly1d(x_6)
    ax2.plot(rd, p_6(rd), label="Least Squares Fit")
    ax2.set_title("Degree=6")

    ax3 = plt.subplot(223)
    ax3.plot(housing[:,0], housing[:,1], 'k*', label="Data Points")
    p_9 = np.poly1d(x_9)
    ax3.plot(rd, p_9(rd), label="Least Squares Fit")
    ax3.set_title("Degree=9")

    ax4 = plt.subplot(224)
    ax4.plot(housing[:,0], housing[:,1], 'k*', label="Data Points")
    p_12 = np.poly1d(x_12)
    ax4.plot(rd, p_12(rd), label="Least Squares Fit")
    ax4.set_title("Degree=12")

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    pts = np.load('ellipse.npy')
    b = np.ones(len(pts))
    A = np.column_stack((pts[:,0]**2, pts[:,0], pts[:,0]*pts[:,1], pts[:,1], pts[:,1]**2))
    x = la.lstsq(A, b)[0]

    plt.ion()
    plt.plot(pts[:,0], pts[:,1], 'k*')
    plot_ellipse(x[0], x[1], x[2], x[3], x[4])

# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape
    x = np.random.random(n)
    x = x/la.norm(x)
    for k in range(N):
        y = x.copy()
        x = A@x
        x = x/la.norm(x)
        if la.norm(x-y) < tol:
            break

    return x@A@x, x


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = A.shape
    S = la.hessenberg(A)
    for k in range(N):
        Q,R = la.qr(S)
        S = R@Q
    eigs = []
    i = 0
    while i < n:
        if i == len(S)-1 or S[i+1][i] < tol:
            eigs.append(S[i][i])
        else:
            b = -(S[i][i]+S[i+1][i+1])
            c = la.det(S[i:i+2,i:i+2])
            eigs.append((-b + cmath.sqrt(b**2 - 4*c))/2)
            eigs.append((-b - cmath.sqrt(b**2 - 4*c))/2)
            i = i+1
        i = i+1

    return eigs


