#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>

#define HEAVY  100000
#define SIZE   30
#define TAG_WORK 1
#define TAG_DONE 2

double heavy(int x, int y) {
	int i, loop;
	double sum = 0;
	if ((x == 3 && y == 3) || (x == 3 && y == 5) ||
		(x == 3 && y == 7) || (x == 20 && y == 10))
		loop = 200;
	else
		loop = 1;
	for (i = 1; i < loop * HEAVY; i++)
		sum += cos(exp(cos((double)i / HEAVY)));
	return sum;
}

// --- Shared task queue for dynamic scheduling ---



typedef struct {
	int row;
	int start_col;
	int end_col;
} ThreadArg;


void* thread_work(void* arg) {
	ThreadArg* t = (ThreadArg*)arg;
	for (int y = t->start_col; y < t->end_col; y++) {
		heavy(t->row, y);
	}
	return NULL;
}


int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
	if (size == 1) {
		const int NUM_THREADS = 4;
		pthread_t threads[NUM_THREADS];
		ThreadArg args[NUM_THREADS];
	
		start_time = MPI_Wtime();  // ⏱️ بدء التوقيت
	
		for (int row = 0; row < SIZE; row++) {
			int cols_per_thread = SIZE / NUM_THREADS;
	
			for (int i = 0; i < NUM_THREADS; i++) {
				args[i].row = row;
				args[i].start_col = i * cols_per_thread;
				args[i].end_col = (i == NUM_THREADS - 1) ? SIZE : args[i].start_col + cols_per_thread;
				pthread_create(&threads[i], NULL, thread_work, &args[i]);
			}
	
			for (int i = 0; i < NUM_THREADS; i++) {
				pthread_join(threads[i], NULL);
			}
		}
	
		double end_time = MPI_Wtime();  // ⏱️ انتهاء التوقيت
		printf("Execution time with 1 node and threads: %f seconds\n", end_time - start_time);
		return 0;
	}
	
	if (rank == 0)
	{
		// Master
		start_time = MPI_Wtime();
		int next_row = 0;
		int active_workers = size - 1;

		while (active_workers > 0) {
			int worker_rank;
			MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK, MPI_COMM_WORLD, &status);

			if (next_row < SIZE) {
				// Send row index to worker
				MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
				next_row++;
			} else {
				// No more work, send termination signal
				int terminate = -1;
				MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD);
				active_workers--;
			}
		}

		end_time = MPI_Wtime();
		printf("Total execution time: %f seconds\n", end_time - start_time);
	}
    else
    {
        const int NUM_THREADS = 24;
        pthread_t threads[NUM_THREADS];
        ThreadArg args[NUM_THREADS];
		// Worker
		while (1) {
			// Request work
			MPI_Send(&rank, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD);

			// Receive task
			int row;
			MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if (status.MPI_TAG == TAG_DONE) {
				break; // No more work
			}

			// Do work
			if (row >= SIZE) break;

            // Distribute the columns of my_row to threads
            int cols_per_thread = SIZE / NUM_THREADS;
            for (int i = 0; i < NUM_THREADS; i++) {
                args[i].row = row;
                args[i].start_col = i * cols_per_thread;
                args[i].end_col = (i == NUM_THREADS - 1) ? SIZE : args[i].start_col + cols_per_thread;
                pthread_create(&threads[i], NULL, thread_work, &args[i]);
            }

            for (int i = 0; i < NUM_THREADS; i++) {
                pthread_join(threads[i], NULL);
            }
            // Notify master that work is done

		}
	}
    // Clean up

    MPI_Finalize();
    return 0;
}
