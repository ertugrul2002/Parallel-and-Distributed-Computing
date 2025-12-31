#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_N 1000
#define NUM_THREADS 2


double calc_log_sum(int **A, int K, int i, int j) {
    double log_sum = 0.0;
    for (int r = i; r < i + K; r++) {
        for (int c = j; c < j + K; c++) {
            if (A[r][c] % 2 == 1 && A[r][c] > 0) {
                log_sum += log(A[r][c]);
            }
        }
    }
    return log_sum;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time = MPI_Wtime();

    int N = 1000, M = 1000, K = 60; 
    int **A = NULL;
    double max_log_sum = -1.0;
    int best_i = 0, best_j = 0;

    if (size != 2 || K > (N/2)) 
    {
        if (rank == 0)
        {
            printf("This program must be run with exactly 2 processes.\n");
        }    
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        srand(time(NULL) + 1);
        A = (int **)malloc(N * sizeof(int *));
        for (int i = 0; i < N; i++) {
            // printf("[");
            A[i] = (int *)malloc(M * sizeof(int));
            for (int j = 0; j < M; j++) {
                A[i][j] = rand() % 100 + 1; 
                // printf("%d ,", A[i][j]);
            }
            // printf("]\n");
        }

    
        for (int i = N / 2; i < N; i++) {
            MPI_Send(A[i], M, MPI_INT, 1, 100 + i, MPI_COMM_WORLD);
        }
    } else {
       
        A = (int **)malloc(N * sizeof(int *));
        for (int i = 0; i < N; i++) {
            A[i] = (int *)malloc(M * sizeof(int));
            if (i >= N / 2) {
                MPI_Recv(A[i], M, MPI_INT, 0, 100 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    int start_row = (rank == 0) ? 0 : N / 2;
    int end_row = (rank == 0) ? (N / 2) : N - K + 1;


    #pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for (int i = start_row; i < end_row ; i++) {
        for (int j = 0; j <= M - K; j++) {
            double log_sum = calc_log_sum(A, K, i, j);
            // printf("Process %d: Submatrix starting at (%d, %d) has log sum = %.4f\n", rank, i, j, log_sum);
            #pragma omp critical
            {
                if (log_sum > max_log_sum) {
                    max_log_sum = log_sum;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }

    if (rank == 1) {
     
        double data[3] = {max_log_sum, (double)best_i, (double)best_j};
        MPI_Send(data, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
       
        double other_data[3];
        MPI_Recv(other_data, 3, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (other_data[0] > max_log_sum) {
            max_log_sum = other_data[0];
            best_i = (int)other_data[1];
            best_j = (int)other_data[2];
        }
        double end_time = MPI_Wtime();
        double execution_time = end_time - start_time;
        printf("Execution time: %.4f seconds\n", execution_time);
        printf("Best submatrix starts at (%d, %d) with log sum = %.4f\n", best_i, best_j, max_log_sum);
    }

    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);

    MPI_Finalize();
    return 0;
}