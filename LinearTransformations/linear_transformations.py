# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
from math import pi
import time

# Problem 1
def stretch(A, a, b):
    """Scale the points in 'A' by 'a' in the x direction and 'b' in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    M = np.array([[a,0],[0,b]])
    return M@A

def shear(A, a, b):
    """Slant the points in 'A' by 'a' in the x direction and 'b' in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    M = np.array([[1,a],[b,1]])
    return M@A

def reflect(A, a, b):
    """Reflect the points in 'A' about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    M = np.array([[a**2 - b**2, 2*a*b],[2*a*b, b**2 - a**2]])
    M = (1/(a**2+b**2))*M
    return M@A

def rotate(A, theta):
    """Rotate the points in 'A' about the origin by 'theta' radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return M@A

# Problem 2
def solar_system(T, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (10,0) and the initial
    position of the moon is (11,0).

    Parameters:
        T (int): The final time.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    pe_0 = np.array([10,0])
    pm_0 = np.array([11,0])

    def pe_t(T, position=pe_0, omega=omega_e):
        return rotate(position, omega*T)

    def pm_t(T, position=pm_0, omega=omega_m):
        return rotate(pm_0-pe_0, omega_m*T) + pe_t(T)

    t_interval = np.linspace(0, T, T*100)
    e_traj1 = [pe_t(t)[0] for t in t_interval]
    e_traj2 = [pe_t(t)[1] for t in t_interval]
    m_traj1 = [pm_t(t)[0] for t in t_interval]
    m_traj2 = [pm_t(t)[1] for t in t_interval]

    plt.ion()
    plt.plot(e_traj1, e_traj2, label="Earth")
    plt.plot(m_traj1, m_traj2, label="Moon")
    plt.axes().set_aspect("equal")
    plt.legend(loc=4, borderaxespad=1)




def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    sizes = [2**i for i in range(9)]
    def timemvp(n):
        x = random_vector(n)
        A = random_matrix(n)
        t1 = time.time()
        matrix_vector_product(A,x)
        t2 = time.time()
        return t2-t1

    def timemmp(n):
        A = random_matrix(n)
        B = random_matrix(n)
        t1 = time.time()
        matrix_matrix_product(A, B)
        t2 = time.time()
        return t2-t1

    time_mv = [timemvp(n) for n in sizes]
    time_mm = [timemmp(n) for n in sizes]

    plt.ion()
    ax1 = plt.subplot(121)
    ax1.plot(sizes, time_mv, 'b.-')
    ax1.set_title("Matrix-Vector Multiplication")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")

    ax2 = plt.subplot(122)
    ax2.plot(sizes, time_mm, 'r.-')
    ax2.set_title("Matrix-Matrix Multiplication")
    ax2.set_xlabel("n")


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    sizes = [2**i for i in range(9)]
    def timemvp(n):
        x = random_vector(n)
        A = random_matrix(n)
        t1 = time.time()
        matrix_vector_product(A,x)
        t2 = time.time()
        return t2-t1

    def timemmp(n):
        A = random_matrix(n)
        B = random_matrix(n)
        t1 = time.time()
        matrix_matrix_product(A, B)
        t2 = time.time()
        return t2-t1

    def npmvp(n):
        x = np.array(random_vector(n))
        A = np.array(random_matrix(n))
        t1 = time.time()
        A@x
        t2 = time.time()
        return t2-t1

    def npmmp(n):
        A,B = np.array(random_matrix(n)), np.array(random_matrix(n))
        t1 = time.time()
        A@B
        t2 = time.time()
        return t2-t1

    time_mv = [timemvp(n) for n in sizes]
    time_mm = [timemmp(n) for n in sizes]
    np_mvp = [npmvp(n) for n in sizes]
    np_mmp = [npmmp(n) for n in sizes]

    plt.ion()
    ax1 = plt.subplot(121)
    ax1.plot(sizes, time_mv, 'b.-', label="Matrix-Vector list op.")
    ax1.plot(sizes, time_mm, 'r.-', label="Matrix-Matrix list op.")
    ax1.plot(sizes, np_mvp, 'g.-', label="Matrix-Vector Numpy op.")
    ax1.plot(sizes, np_mmp, 'c.-', label="Matrix-Matrix Numpy op.")
    ax1.set_title("Linear scale")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(122)
    ax2.loglog(sizes, time_mv, 'b.-', basex=2, basey=2)
    ax2.loglog(sizes, time_mm, 'r.-', basex=2, basey=2)
    ax2.loglog(sizes, np_mvp, 'g.-', basex=2, basey=2)
    ax2.loglog(sizes, np_mmp, 'c.-', basex=2, basey=2)
    ax2.set_title("Logarithmic scale")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Seconds")


#Test functions for problem 1
def teststretch(A, a=1/2, b=6/5):
    plt.ion()
    lim = [-1,1,-1,1]

    ax1 = plt.subplot(121)
    ax1.plot(A[0], A[1], 'k,')
    ax1.set_title("Original")
    plt.axis(lim)

    S = stretch(A, a, b)
    ax2 = plt.subplot(122)
    ax2.plot(S[0], S[1], 'k,')
    ax2.set_title("Stretch")
    plt.axis(lim)

def testshear(A, a=1/2, b=0):
    plt.ion()
    lim = [-1,1,-1,1]

    ax1 = plt.subplot(121)
    ax1.plot(A[0], A[1], 'k,')
    ax1.set_title("Original")
    plt.axis(lim)

    S = shear(A, a, b)
    ax2 = plt.subplot(122)
    ax2.plot(S[0], S[1], 'k,')
    ax2.set_title("Shear")
    plt.axis(lim)

def testreflect(A, a=0, b=1):
    plt.ion()
    lim = [-1,1,-1,1]

    ax1 = plt.subplot(121)
    ax1.plot(A[0], A[1], 'k,')
    ax1.set_title("Original")
    plt.axis(lim)

    S = reflect(A, a, b)
    ax2 = plt.subplot(122)
    ax2.plot(S[0], S[1], 'k,')
    ax2.set_title("Reflection")
    plt.axis(lim)

def testrotate(A, theta=pi/2):
    plt.ion()
    lim = [-1,1,-1,1]

    ax1 = plt.subplot(121)
    ax1.plot(A[0], A[1], 'k,')
    ax1.set_title("Original")
    plt.axis(lim)

    S = rotate(A, theta)
    ax2 = plt.subplot(122)
    ax2.plot(S[0], S[1], 'k,')
    ax2.set_title("Rotation")
    plt.axis(lim)
