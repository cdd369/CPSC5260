#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 100
#define K 32

// Sequential matrix multiplication
void Multiply_serial(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0f;
            for (int k = 0; k < n; k++)
                C[i*p + j] += A[i*n + k] * B[k*p + j];
        }
}

// Check if two matrices are equal
int IsEqual(float *A, float *B, int m, int n) {
    for (int i = 0; i < m*n; i++)
        if (A[i] != B[i])
            return 0;
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float *A = NULL, *B = NULL, *C = NULL, *C_serial = NULL;

    int rows_per_proc = N / size;
    int remainder = N % size;

    // Number of rows handled by this process
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // Allocate memory
    float *A_local = (float*)malloc(local_rows * K * sizeof(float));
    float *C_local = (float*)malloc(local_rows * N * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));

    if (rank == 0) {
        A = (float*)malloc(N * K * sizeof(float));
        C = (float*)malloc(N * N * sizeof(float));
        C_serial = (float*)malloc(N * N * sizeof(float));

        srand(time(NULL));
        for (int i = 0; i < N*K; i++) A[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < K*N; i++) B[i] = (float)rand() / RAND_MAX;

        double t1 = MPI_Wtime();
        Multiply_serial(A, B, C_serial, N, K, N);
        double t2 = MPI_Wtime();
        printf("Sequential multiplication time: %f seconds\n", t2 - t1);
    }

    // Broadcast B
    MPI_Bcast(B, K*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Synchronize before parallel timing starts
    MPI_Barrier(MPI_COMM_WORLD);
    double t_parallel_start = MPI_Wtime();

    // Distribute rows of A
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int send_rows = rows_per_proc + (i < remainder ? 1 : 0);
            int offset = i*rows_per_proc + (i < remainder ? i : remainder);
            MPI_Send(&A[offset*K], send_rows*K, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < local_rows*K; i++) A_local[i] = A[i];
    } else {
        MPI_Recv(A_local, local_rows*K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Local computation
    Multiply_serial(A_local, B, C_local, local_rows, K, N);

    // Gather results
    if (rank == 0) {
        for (int i = 0; i < local_rows*N; i++) C[i] = C_local[i];
        for (int i = 1; i < size; i++) {
            int recv_rows = rows_per_proc + (i < remainder ? 1 : 0);
            int offset = i*rows_per_proc + (i < remainder ? i : remainder);
            MPI_Recv(&C[offset*N], recv_rows*N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(C_local, local_rows*N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }

    // End parallel timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t_parallel_end = MPI_Wtime();
    if (rank == 0)
        printf("Parallel multiplication time (Blocking P2P): %f seconds\n", t_parallel_end - t_parallel_start);

    // Check correctness
    if (rank == 0) {
        if (IsEqual(C, C_serial, N, N))
            printf("Parallel result matches serial computation!\n");
        else
            printf("Parallel result does NOT match serial computation!\n");

        free(A);
        free(C);
        free(C_serial);
    }

    free(A_local);
    free(B);
    free(C_local);

    MPI_Finalize();
    return 0;
}
