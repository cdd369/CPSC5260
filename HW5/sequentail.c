#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

#define N = 1000
#define M = 32
int main(int argc, char** argv)
{
    float A[N][M];
    float B[M][N];
    float C[N][N];

    // Seed the random number generator
    srand(time(NULL));

    // Initialize matrices A and B with random numbers between 0 and 1

    for (int i = 0, i < N; i++){
        for (int j = 0, j<M; j++){
            A[i][j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0, i < M; i++){
        for (int j = 0, j<N; j++){
            A[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // Initialize result matrix C to 0
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }

    // Matrix multiplication A (N x 32) * B (32 x N) = C (N x N)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Print result matrix C
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.4f ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}