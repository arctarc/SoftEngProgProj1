#====

from libc.math cimport cos, sin, exp
import numpy as np
cimport numpy as cnp

# We need to add a few more imports for OpenMP to work

from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
cimport openmp

#====

# one_energy function: (This remains unchanged as there are no loops here)

cpdef double one_energy(cnp.ndarray[cnp.float64_t, ndim=2] arr, int ix, int iy, int nmax):
    
    cdef double en = 0.0
    cdef int ixp, ixm, iyp, iym
    cdef double ang

    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    return en

#====

# all_energy function: (This remains unchanged as all_energy calls another cpdef function, which is not allowed)

cpdef double all_energy(cnp.ndarray[cnp.float64_t, ndim=2] arr, int nmax):
    
    cdef double enall = 0.0
    cdef int i, j

    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
            
    return enall

#====

# get_order function: (This will be parallelised)

cpdef double get_order(cnp.ndarray[cnp.float64_t, ndim=2] arr, int nmax):
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] delta = np.eye(3, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] lab
    cdef int a, b, i, j

    lab = np.zeros((3, nmax, nmax), dtype=np.float64)
    lab[0, :, :] = np.cos(arr)
    lab[1, :, :] = np.sin(arr)

    with nogil:
        for a in range(3):
            for b in range(3):
                for i in prange(nmax, schedule='dynamic', num_threads=openmp.omp_get_max_threads()):
                # Modifications made for parallel loop execution.
                    for j in range(nmax):
                        Qab[a, b] += 3.0 * lab[a, i, j] * lab[b, i, j] - delta[a, b]

    Qab /= (2 * nmax * nmax)

    eigenvalues, _ = np.linalg.eig(Qab)

    return np.max(eigenvalues)

#====

# MC_step function: (This remains unchanged as all_energy calls another cpdef function, which is not allowed)

cpdef double MC_step(cnp.ndarray[cnp.float64_t, ndim=2] arr, double Ts, int nmax):
    
    cdef double scale = 0.1 + Ts
    cdef int accept = 0
    cdef cnp.ndarray[cnp.int_t, ndim=2] xran = np.random.randint(0, nmax, size=(nmax, nmax))
    cdef cnp.ndarray[cnp.int_t, ndim=2] yran = np.random.randint(0, nmax, size=(nmax, nmax))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] aran = np.random.normal(0.0, scale, size=(nmax, nmax))
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)

            if en1 <= en0:
                accept += 1
            else:
                randVal = rand() / <double>RAND_MAX
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= randVal:
                    # Using a randomiser that is safe for OpenMP
                    accept += 1
                else:
                    arr[ix, iy] -= ang

    return accept / (nmax * nmax)

#====