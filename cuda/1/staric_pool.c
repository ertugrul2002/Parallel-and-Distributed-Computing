#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>

#define HEAVY  100000
#define SIZE   30

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

typedef struct {
	int start_x;
	int end_x;
} ThreadArg;

void* thread_work(void* arg) {
	ThreadArg* t = (ThreadArg*)arg;
	for (int x = t->start_x; x < t->end_x; x++) {
		for (int y = 0; y < SIZE; y++) {
			heavy(x, y); // just perform the heavy computation
		}
	}
	return NULL;
}

int main(int argc, char* argv[]) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Barrier(MPI_COMM_WORLD); // sync all processes before timing
	double start_time = MPI_Wtime(); // start timer

	// تقسيم الصفوف على العمليات
	int rows_per_proc = SIZE / size;
	int start_row = rank * rows_per_proc;
	int end_row = (rank == size - 1) ? SIZE : start_row + rows_per_proc;

	const int NUM_THREADS = 24;
	pthread_t threads[NUM_THREADS];
	ThreadArg args[NUM_THREADS];

	int rows_per_thread = (end_row - start_row) / NUM_THREADS;

	for (int i = 0; i < NUM_THREADS; i++) {
		args[i].start_x = start_row + i * rows_per_thread;
		args[i].end_x = (i == NUM_THREADS - 1) ? end_row : args[i].start_x + rows_per_thread;
		pthread_create(&threads[i], NULL, thread_work, &args[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	MPI_Barrier(MPI_COMM_WORLD); // sync again before ending timer
	double end_time = MPI_Wtime(); // end timer

	double local_elapsed = end_time - start_time;
	double max_elapsed;

	MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Total execution time: %f seconds\n", max_elapsed);
	}

	MPI_Finalize();
	return 0;
}
