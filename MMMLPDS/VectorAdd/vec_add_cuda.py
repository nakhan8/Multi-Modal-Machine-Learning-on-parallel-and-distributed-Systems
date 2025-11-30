from numba import njit, prange
import numpy as np
import time

N = 10_000_000
A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)

@njit(parallel=True)
def vec_add(A, B):
    C = np.empty_like(A)
    for i in prange(A.size):
        C[i] = A[i] + B[i]
    return C

start = time.time()
C = vec_add(A, B)
end = time.time()

print("Elapsed time:", end - start)
print("C[0] =", C[0], "C[N-1] =", C[-1])
