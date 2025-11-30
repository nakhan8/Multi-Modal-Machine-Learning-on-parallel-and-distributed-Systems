from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10_000_000
local_n = N // size

if rank == 0:
    A = np.arange(N, dtype=np.float64)
    B = np.ones(N, dtype=np.float64)
else:
    A = None; B = None

localA = np.empty(local_n, dtype=np.float64)
localB = np.empty(local_n, dtype=np.float64)

comm.Barrier()  # sync before timing
start = MPI.Wtime()

comm.Scatter([A, MPI.DOUBLE], [localA, MPI.DOUBLE], root=0)
comm.Scatter([B, MPI.DOUBLE], [localB, MPI.DOUBLE], root=0)

localC = localA + localB

if rank == 0:
    C = np.empty(N, dtype=np.float64)
else:
    C = None

comm.Gather([localC, MPI.DOUBLE], [C, MPI.DOUBLE], root=0)

comm.Barrier()
end = MPI.Wtime()

if rank == 0:
    print("C[0], C[-1] =", C[0], C[-1])
    print("MPI runtime with {} processes: {:.6f} seconds".format(size, end - start))
