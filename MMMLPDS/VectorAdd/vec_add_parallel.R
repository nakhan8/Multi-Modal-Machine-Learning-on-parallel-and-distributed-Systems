# vec_add_parallel.R
library(parallel)

n <- 10000000
a <- rep(1, n)
b <- 1:n
c <- numeric(n)

# Number of cores
num_cores <- detectCores()
cl <- makeCluster(num_cores)

start_time <- Sys.time()

chunk_size <- ceiling(n / num_cores)
results <- parLapply(cl, 0:(num_cores-1), function(i){
  s <- i*chunk_size + 1
  e <- min((i+1)*chunk_size, n)
  c_chunk <- numeric(e-s+1)
  for (j in s:e) {
    c_chunk[j-s+1] <- a[j] + b[j]
  }
  c_chunk
})

# Combine results
c <- unlist(results)
stopCluster(cl)

end_time <- Sys.time()
elapsed <- end_time - start_time
cat("c[0] =", c[1], ", c[n-1] =", c[n], ", elapsed =", elapsed, "seconds\n")
