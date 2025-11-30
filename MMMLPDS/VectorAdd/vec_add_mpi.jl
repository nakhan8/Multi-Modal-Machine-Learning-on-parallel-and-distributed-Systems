# vec_add_mpi.jl
using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

N = 1000000
local_n = div(N, size)

if rank==0
    A = collect(0.0:(N-1))
    B = ones(Float64, N)
else
    A = nothing; B = nothing
end

localA = zeros(Float64, local_n)
localB = zeros(Float64, local_n)

MPI.Scatter!(A, localA, 0, comm)
MPI.Scatter!(B, localB, 0, comm)

localC = localA .+ localB

if rank==0
    C = zeros(Float64, N)
else
    C = nothing
end

MPI.Gather!(localC, C, 0, comm)

if rank==0
    println(C[1], " ", C[end])
end

MPI.Finalize()