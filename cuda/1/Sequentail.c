#include <stdio.h>
#include <math.h>
#include <time.h>

#define HEAVY  100000
#define SIZE   30

// This function performs heavy computations
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

int main(int argc, char* argv[]) {
	int x, y;
	int size = SIZE;
	double answer = 0;

	// ابدأ الوقت
	clock_t start = clock();

	for (x = 0; x < size; x++)
		for (y = 0; y < size; y++)
			answer += heavy(x, y);

	// احسب الزمن المنقضي
	clock_t end = clock();
	double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

	printf("answer = %e\n", answer);
	printf("Execution time: %f seconds\n", time_taken);

	return 0;
}
