// File: vec_add_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s <vector_size>\n", argv[0]);
        return 1;
    }

    long n = atol(argv[1]);
    double *a = (double*) malloc(n * sizeof(double));
    double *b = (double*) malloc(n * sizeof(double));
    double *c = (double*) malloc(n * sizeof(double));

    // Initialize vectors
    #pragma omp parallel for
    for(long i = 0; i < n; i++) {
        a[i] = i + 1.0;
        b[i] = i + 1.0;
    }

    double start = omp_get_wtime();

    // Vector addition
    #pragma omp parallel for
    for(long i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    double end = omp_get_wtime();

    printf("c[0] = %f, c[n-1] = %f, elapsed = %f seconds\n", c[0], c[n-1], end - start);

    free(a);
    free(b);
    free(c);

    return 0;
}
