// Vinnie Zhang
// Parallel Computing - Lab 2

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


int main(int argc, char *argv[]) {

	char *N = argv[1]; // positive number greater than and less than or equal to 100,000 
	char *t = argv[2]; // number of threads and is a positive int <= 100
	int num = atoi(N); // converting from string to int
	int thread = atoi(t); // converting from string to int
	int stop = (num + 1) / 2;

	// instantiating variables
	int prime[num - 1];
	double time_start = 0.0, time_taken;

	time_start = omp_get_wtime();

	#pragma omp parallel num_threads (thread)
	{

		#pragma omp for
		for (int i = 2; i <= num; i++) {
			prime[i-2] = i;
		}

		#pragma omp for 
		for (int j = 2; j <= stop; j++) {
			
			if (prime[j-2] == 0) {
				continue;
			
			}

			for (int k = j-1; k < num-1; k++) {
				if (prime[k] % j == 0) {
					prime[k] = 0;
				}
			}
		}	
	}

	time_taken = omp_get_wtime() - time_start; // calculating time elapsed from start
	printf("Time take for the main part: %f\n", time_taken); // printing time elapsed

	// writing to file
	FILE *fp;
	fp = fopen ("N.txt", "w");
	int previous_prime = 2;
	int index = 1;

	for (int k = 0; k < num-1; k++) {

		if (prime[k] != 0) {

			fprintf(fp, "%d, %d, %d\n", index, prime[k], prime[k]-previous_prime);
			index += 1;
			previous_prime = prime[k];
		}
	}

	fclose(fp);
}