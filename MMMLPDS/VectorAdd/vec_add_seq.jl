# vec_add_seq.jl
n = 10_000_000
a = ones(Float32, n)
b = collect(1.0f0:n)
c = similar(a)

# measure time
@time begin
    for i in 1:n
        c[i] = a[i] + b[i]
    end
end

println("c[0] = ", c[1], ", c[n-1] = ", c[end])
