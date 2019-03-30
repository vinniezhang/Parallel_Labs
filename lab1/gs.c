// Parallel Computing Lab 1
// Vinnie Zhang 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/


/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]); // fabs() function indicates absolute value of a[i][i] matrix
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum) // matrix doesn't converge if aii < sum
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r"); // opening file
  
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num); // number of unknowns
 fscanf(fp,"%f ",&err); // relative error

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float));

 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 


 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); // closing file

}


/************************************************************/


int main(int argc, char *argv[])
{

 int i, j;
 int nit = 0; /* number of iterations */
 FILE * fp;
 char output[100] ="";
  
 if( argc != 2)
 {
   printf("Usage: ./gsref filename\n");
   exit(1);
 }
  
 /* Read the input file and fill the global data structure above */ 
 get_input(argv[1]);
 
 /* Check for convergence condition */
 /* This function will exit the program if the coffeicient will never converge to 
  * the needed absolute error. 
  * This is not expected to happen for this programming assignment.
  */
 check_matrix();

 float *new_x;
 float *new_x_sum; // array of all of the x values?
 int complete = 0; // to indicate if the rel error is <= given rel error
 int local_complete; // indicates completion of processes

// initiating arrays 
 new_x = (float *) calloc(num, sizeof(float));
 new_x_sum = (float *) malloc(num * sizeof(float));

 int comm_sz; // number of processes in the group
 int my_rank; // rank of current process
 int count;
 int first_index; // of current process
 int last_index; // of current porcess
 
 MPI_Init(&argc, &argv); // initializing with given arguments
 MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); // determines size of the group (# of processes)
 MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // determines rank of the calling process in the communicator

// variable declarations for later equation calculation
 int quotient = num / comm_sz; // unknowns / processes
 int remainder = num % comm_sz;

// declaration of variables used to keep track of time
// double start, finish; // local start and finish times of processes
// double overhead = 0; // for tracking overhead time
// double begin, end; // used to calculate overhead time

 if (my_rank < remainder) {
 	count = quotient++;
 	first_index = my_rank*count;
 }

 else {
 	count = quotient;
 	first_index = my_rank*count + remainder;
 }

last_index = first_index + count;

// do this while the rel error is not <= the given rel error
while (complete == 0){ 

	local_complete = 1; // an iteration has completed
	nit += 1; // iteration count increments

	MPI_Barrier(MPI_COMM_WORLD);

	// begin = MPI_Wtime(); // starting local timer

	// implementation of the matrices into C code
	for (int i = first_index; i < last_index; i++)
	{
		printf("Process %d now updates index %d\n", my_rank, i);
      	new_x[i] = b[i] + a[i][i]*x[i];
      
      	for (j = 0; j < num; j++) {
       		new_x[i] -= a[i][j]*x[j];
      	}

      	new_x[i] /= a[i][i];
     
         if ((new_x[i] - x[i]) / new_x[i] > err)
           local_complete = 0;
    } 

    // end = MPI_Wtime(); // ending local timer
	MPI_Barrier(MPI_COMM_WORLD); // blocks until all processes have reached this end
	// start = MPI_Wtime(); // starting clock
	MPI_Allreduce(new_x, new_x_sum, num, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_complete, &complete, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
	// finish = MPI_Wtime(); // end timer
	// overhead += finish - start;

	for (i = 0; i < num; i++){
		x[i] = new_x_sum[i];
	}

	MPI_Barrier(MPI_COMM_WORLD);
 	MPI_Finalize();

}

 /* Writing results to file */
 sprintf(output,"%d.sol",num);
 fp = fopen(output,"w");

 if(!fp)
 {
   printf("Cannot create the file %s\n", output);
   exit(1);
 }
    
 for( i = 0; i < num; i++)
   fprintf(fp,"%f\n",x[i]);
 
 printf("total number of iterations: %d\n", nit);
 
 fclose(fp);
 
 exit(0);


}
