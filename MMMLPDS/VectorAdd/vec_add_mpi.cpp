#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start = MPI_Wtime();

    const int N = 1000000;
    int local_n = N / size;

    std::vector<float> A(local_n), B(local_n), C(local_n);
    std::vector<float> fullA, fullB, fullC;

    if (rank == 0) {
        fullA.resize(N, 1.0f);
        fullB.resize(N, 1.0f);
    }

    MPI_Scatter(fullA.data(), local_n, MPI_FLOAT, A.data(), local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(fullB.data(), local_n, MPI_FLOAT, B.data(), local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; ++i)
        C[i] = A[i] + B[i];

    if (rank == 0) fullC.resize(N);
    MPI_Gather(C.data(), local_n, MPI_FLOAT, fullC.data(), local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Total runtime (MPI, including scatter/gather): " << (t_end - t_start) << " s" << std::endl;
        std::cout << "C[0] = " << fullC[0] << ", C[N-1] = " << fullC[N-1] << std::endl;
    }

    MPI_Finalize();
}
