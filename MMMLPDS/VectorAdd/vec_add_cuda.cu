// vec_add_cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vec_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    const int N = 1000000;
    size_t size = N * sizeof(float);

    float *hA = new float[N], *hB = new float[N], *hC = new float[N];
    for (int i = 0; i < N; ++i) hA[i] = hB[i] = 1.0f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    vec_add_kernel<<<blocks_per_grid, threads_per_block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    auto t_end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Total runtime (CUDA, including memory copy): " << runtime << " s" << std::endl;
    std::cout << "C[0] = " << hC[0] << ", C[N-1] = " << hC[N-1] << std::endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
}