#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N  10
#define M  32

void Matrix_Multiply(double *A, double *B, double*C, int m, int n, int p)
{
    int i, j, k;
    for (i = 0; i<m;i++)
        for (j = 0; j < p; j++)
        {
            C[i*p + j] = 0.0f;
            for (k = 0; k < n; k++)
                C[i*p + j] += A[i*n + k] * B[k*p + j];
        }
}

int IsEqual(double *A, double *B, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (A[i*n + j] != B[i*n + j])
            {
                return 0; // Not equal
            }
        }
    }
    return 1; // Equal
}

int main(int argc, char** argv)
{ 
	int rank,size;
	double *A = NULL, *B = NULL;
	double *local_C = (double*)malloc(N*N*sizeof(double));
	double *global_C = (double*)malloc(N*N*sizeof(double));
	
	MPI_Init(&argc, &agrv)
	MPI_Comm_rank(MPI_COMM_WORLD, &rank)
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	// Each process gets n/size elements
	int n = N*M;
	int local_n = n/size;
	double *local_A  = (double*)malloc(local_n * sizeof(double));
    double *local_B = (double*)malloc(local_n * sizeof(double));
	
	    // Root process initializes data
    if (rank == 0) {
        A = (double*)malloc(n * sizeof(double));
        B = (double*)malloc(n * sizeof(double));

        srand(time(NULL));

		// Initialize A
		for (int i = 0; i < N*M; i++) {
			A[i] = (double)rand() / RAND_MAX;
			printf("%8.4f ", A[i]);
		}

		// Initialize B
		for (int i = 0; i < M*N; i++) {
			B[i] = (double)rand() / RAND_MAX;
			printf("%8.4f ", A[i]);
		}
        printf("Vectors initialized.\n");
    }
	

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }
	
	// Scatter the data to all processes
    MPI_Scatter(A, local_n, MPI_DOUBLE, local_A, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_n, MPI_DOUBLE, local_B, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    srand(time(NULL));
	
    local_C = Matrix_Multiply(local_A, local_B, local_C, N, M, N);

    // Print result matrix C
    printf("Result matrix C:\n");
    for (int i = 0; i <N*N; i++) {
            printf("%8.4f ", C[i]);
    }
    
    //Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}

