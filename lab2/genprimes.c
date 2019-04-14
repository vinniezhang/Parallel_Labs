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
	int stop = (num + 1) / 2; // number to stop at (right before the last specified prime number)

	// instantiating variables
	int primes[num - 1]; // array of prime numbers
	double time_start = 0.0, time_taken;
	time_start = omp_get_wtime();
	char output[100] =""; // for outputting result

	#pragma omp parallel num_threads (thread) // parallelizing processes
	{

		#pragma omp for
		for (int i = 2; i <= num; i++) {
			primes[i-2] = i;
		}

		#pragma omp for 
		for (int j = 2; j <= stop; j++) {
			
			if (primes[j-2] == 0) {
				continue;
			
			}

			for (int k = j-1; k < num-1; k++) {
				if (primes[k] % j == 0) {
					primes[k] = 0;
				}
			}
		}	
	}

	time_taken = omp_get_wtime() - time_start; // calculating time elapsed from start
	printf("Time take for the main part: %f\n", time_taken); // printing time elapsed

	// writing results to file
	FILE *fp;
	sprintf(output,"%d.txt",num);
	fp = fopen (output, "w");
	
	int previous_prime = 2;
	int index = 1;

	for (int k = 0; k < num-1; k++) {

		if (primes[k] != 0) {

			fprintf(fp, "%d, %d, %d\n", index, primes[k], primes[k]-previous_prime);
			index += 1;
			previous_prime = primes[k];
		}
	}

	fclose(fp);
}