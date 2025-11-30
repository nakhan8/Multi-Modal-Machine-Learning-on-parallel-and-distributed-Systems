# vec_add_seq.R
n <- 10000000
a <- rep(1, n)
b <- 1:n
c <- numeric(n)

start_time <- Sys.time()
for (i in 1:n) {
  c[i] <- a[i] + b[i]
}
end_time <- Sys.time()

elapsed <- end_time - start_time
cat("c[0] =", c[1], ", c[n-1] =", c[n], ", elapsed =", elapsed, "seconds\n")