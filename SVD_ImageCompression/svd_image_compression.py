# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    eigs, V = la.eig(A.conj().T @ A)
    V = V.T
    eigs, V = eigs[np.where(eigs > tol)], V[np.where(eigs > tol)]
    s_values = np.array([np.sqrt(np.real(ev)) for ev in eigs])
    sorted_s_values = np.sort(s_values)[::-1]
    indices = []
    for i in range(len(s_values)):
    	for j in range(len(s_values)):
    		if sorted_s_values[i] == s_values[j]:
    			indices.append(j)
    V = V[np.array(indices)]
    U = A@V.T/sorted_s_values
    return U, sorted_s_values, V

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    S1 = np.array([np.cos(theta) for theta in np.linspace(0,2*np.pi,200)])
    S2 = np.array([np.sin(theta) for theta in np.linspace(0,2*np.pi,200)])
    S = np.vstack([S1,S2])
    E = np.array([[1,0,0],[0,0,1]])
    plt.ion()
    ax1 = plt.subplot(221)
    ax1.plot(S[0], S[1])
    ax1.plot(E[0], E[1])
    ax1.axis("equal")

    U, s, Vh = la.svd(A)
    VhS = Vh@S
    VhE = Vh@E
    ax2 = plt.subplot(222)
    ax2.plot(VhS[0], VhS[1])
    ax2.plot(VhE[0], VhE[1])
    ax2.axis("equal")


    sVhS = np.diag(s)@VhS
    sVhE = np.diag(s)@VhE
    ax3 = plt.subplot(223)
    ax3.plot(sVhS[0], sVhS[1])
    ax3.plot(sVhE[0], sVhE[1])
    ax3.axis("equal")

    UsVhS = U@sVhS
    UsVhE = U@sVhE
    ax4 = plt.subplot(224)
    ax4.plot(UsVhS[0], UsVhS[1])
    ax4.plot(UsVhE[0], UsVhE[1])
    ax4.axis("equal")


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    if s > np.linalg.matrix_rank(A):
    	raise ValueError("s is greater than the rank of A")
    U, S, Vh = la.svd(A, full_matrices=False)
    U = U[:,:s]
    S = S[:s]
    Vh = Vh[:s]
    A = U@np.diag(S)@Vh
    return A, U.size+S.size+Vh.size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, S, Vh = la.svd(A, full_matrices=False)
    s = len(np.where(S > err)[0])
    if len(s) == len(S):
    	raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank.")
    U = U[:,:s]
    S = S[:s]
    Vh = Vh[:s]
    A = U@np.diag(S)@Vh
    return A, U.size+S.size+Vh.size

# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image foplund at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = plt.imread(filename) / 255
    plt.ion()

    if len(image.shape) == 2:
    	A, new_size = svd_approx(image, s)
    	ax1 = plt.subplot(121)
    	ax1.imshow(image, cmap="gray")
    	plt.axis("off")
    	ax2 = plt.subplot(122)
    	ax2.imshow(A, cmap="gray")
    	plt.axis("off")
    	plt.suptitle(str(image.size-new_size))

    else:
    	Rs, r_size = svd_approx(image[:,:,0], s)
    	Gs, g_size = svd_approx(image[:,:,1], s)
    	Bs, b_size = svd_approx(image[:,:,2], s)
    	Rs = np.clip(Rs, 0, 1)
    	Gs = np.clip(Gs, 0, 1)
    	Bs = np.clip(Bs, 0, 1)
    	A = np.dstack([Rs, Gs, Bs])
    	ax1 = plt.subplot(121)
    	ax1.imshow(image)
    	plt.axis("off")
    	ax2 = plt.subplot(122)
    	ax2.imshow(A)
    	plt.axis("off")
    	plt.suptitle(str(image.size-(r_size+g_size+b_size)))








