import numpy as np
from numba import njit, prange, set_num_threads
import time

@njit(parallel=True)
def vec_add(A,B):
    N = A.shape[0]
    C = np.empty(N, dtype=A.dtype)
    for i in prange(N):
        C[i] = A[i] + B[i]
    return C

N = 10_000_000
A = np.arange(N, dtype=np.float64)
B = np.ones(N, dtype=np.float64)

set_num_threads(8)

start = time.time()
C = vec_add(A, B)
end = time.time()

print("C[0], C[-1] =", C[0], C[-1])
print("Numba runtime: {:.6f} seconds".format(end - start))
