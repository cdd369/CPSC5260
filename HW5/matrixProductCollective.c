#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 1000
#define K 32

void Multiply_serial(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0f;
            for (int k = 0; k < n; k++)
                C[i*p + j] += A[i*n + k] * B[k*p + j];
        }
}

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

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int *recvcounts = malloc(size * sizeof(int));
    int *recvdispls = malloc(size * sizeof(int));

    int offset = 0;
    for(int i = 0; i < size; i++){
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * K;
        displs[i] = offset * K;
        recvcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
        recvdispls[i] = offset * N;
        offset += rows_per_proc + (i < remainder ? 1 : 0);
    }

    int local_rows = sendcounts[rank] / K;

    float *A_local = malloc(local_rows * K * sizeof(float));
    float *C_local = malloc(local_rows * N * sizeof(float));
    B = malloc(K * N * sizeof(float));

    if(rank == 0){
        A = malloc(N * K * sizeof(float));
        C = malloc(N * N * sizeof(float));
        C_serial = malloc(N * N * sizeof(float));

        srand(time(NULL));
        for(int i=0;i<N*K;i++) A[i]=(float)rand()/RAND_MAX;
        for(int i=0;i<K*N;i++) B[i]=(float)rand()/RAND_MAX;

        double t1 = MPI_Wtime();
        Multiply_serial(A,B,C_serial,N,K,N);
        double t2 = MPI_Wtime();
        printf("Sequential multiplication time: %f sec\n", t2-t1);
    }

    MPI_Bcast(B, K*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT,
                 A_local, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    double t3 = MPI_Wtime();
    Multiply_serial(A_local, B, C_local, local_rows, K, N);
    double t4 = MPI_Wtime();

    MPI_Gatherv(C_local, recvcounts[rank], MPI_FLOAT,
                C, recvcounts, recvdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Parallel (Collective) multiplication time: %f sec\n", t4 - t3);

        if(IsEqual(C, C_serial, N,N))
            printf("Result matches!\n");
        else
            printf("Result mismatch!\n");

        free(A); free(C); free(C_serial);
    }

    free(A_local); free(B); free(C_local);
    free(sendcounts); free(displs); free(recvcounts); free(recvdispls);

    MPI_Finalize();
    return 0;
}
