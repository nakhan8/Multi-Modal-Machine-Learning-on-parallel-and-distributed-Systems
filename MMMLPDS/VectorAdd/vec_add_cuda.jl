# vec_add_cuda.jl
using CUDA
N = 10_000_000
A = CuArray(collect(0f0:(N-1)))
B = CuArray(ones(Float32, N))
C = similar(A)
function kernel!(C, A, B)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds C[i] = A[i] + B[i]
    end
    return
end

@cuda threads=256 blocks=cld(N,256) kernel!(C,A,B)
synchronize()
hC = Array(C)
println(hC[1], " ", hC[end])