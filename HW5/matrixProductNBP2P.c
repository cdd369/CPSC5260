#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

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

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    float *A_local = (float*) malloc(local_rows * K * sizeof(float));
    float *C_local = (float*) malloc(local_rows * N * sizeof(float));
    B = (float*) malloc(K * N * sizeof(float));

    if(rank == 0){
        A = (float*) malloc(N * K * sizeof(float));
        C = (float*) malloc(N * N * sizeof(float));
        C_serial = (float*) malloc(N * N * sizeof(float));

        srand(time(NULL));
        for(int i=0;i<N*K;i++) A[i]=(float)rand()/RAND_MAX;
        for(int i=0;i<K*N;i++) B[i]=(float)rand()/RAND_MAX;

        double t1 = MPI_Wtime();
        Multiply_serial(A, B, C_serial, N, K, N);
        double t2 = MPI_Wtime();
        printf("Sequential multiplication time: %f seconds\n", t2 - t1);
    }

    MPI_Bcast(B, K*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Request req_send_A[1], req_recv_A[1];

    if(rank == 0){
        for(int i = 1; i < size; i++){
            int send_rows = rows_per_proc + (i < remainder ? 1 : 0);
            int offset = i*rows_per_proc + (i < remainder ? i : remainder);
            MPI_Isend(&A[offset*K], send_rows*K, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req_send_A[0]);
        }
        memcpy(A_local, A, local_rows*K*sizeof(float));
    } else {
        MPI_Irecv(A_local, local_rows*K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req_recv_A[0]);
        MPI_Wait(&req_recv_A[0], MPI_STATUS_IGNORE);
    }

    double t3 = MPI_Wtime();
    Multiply_serial(A_local, B, C_local, local_rows, K, N);
    double t4 = MPI_Wtime();

    MPI_Request req_send_C[1], req_recv_C[1];

    if(rank == 0){
        memcpy(&C[start_row*N], C_local, local_rows*N*sizeof(float));

        for(int i=1;i<size;i++){
            int recv_rows = rows_per_proc + (i < remainder ? 1 : 0);
            int offset = i*rows_per_proc + (i < remainder ? i : remainder);
            MPI_Irecv(&C[offset*N], recv_rows*N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &req_recv_C[0]);
            MPI_Wait(&req_recv_C[0], MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Isend(C_local, local_rows*N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req_send_C[0]);
        MPI_Wait(&req_send_C[0], MPI_STATUS_IGNORE);
    }

    if(rank == 0){
        printf("Parallel (Non-Blocking P2P) time: %f seconds\n", t4 - t3);
        if(IsEqual(C, C_serial, N, N))
            printf("Parallel result MATCHES serial result.\n");
        else
            printf("Parallel result does NOT match serial result.\n");

        free(A); free(C); free(C_serial);
    }

    free(A_local); free(B); free(C_local);
    MPI_Finalize();
    return 0;
}
