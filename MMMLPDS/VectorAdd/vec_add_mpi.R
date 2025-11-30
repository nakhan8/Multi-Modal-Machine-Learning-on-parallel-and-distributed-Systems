# vec_add_mpi.R
library(Rmpi)
mpi.spawn.Rslaves()
comm.size <- mpi.comm.size()
rank <- mpi.comm.rank()

N <- 1000000
local_n <- N / comm.size

if(rank==0){
  A <- as.double(0:(N-1))
  B <- rep(1.0, N)
} else {
  A <- NULL; B <- NULL
}

localA <- mpi.scatter.Robj(A, local_n, root=0)
localB <- mpi.scatter.Robj(B, local_n, root=0)

localC <- localA + localB

C <- mpi.gather.Robj(localC, root=0)
if(rank==0) print(c(C[1], C[length(C)]))

mpi.close.Rslaves()
mpi.quit()
