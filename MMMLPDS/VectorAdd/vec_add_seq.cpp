#include <iostream>
#include <vector>
#include <chrono>

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    const int N = 1000000;
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 1.0f);
    std::vector<float> C(N);

    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];

    auto t_end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Total runtime (CPU): " << runtime << " s\n";
    std::cout << "C[0] = " << C[0] << ", C[N-1] = " << C[N-1] << std::endl;
}
