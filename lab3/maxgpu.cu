// Vinnie Zhang
// Parallel Computing - Lab 3

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


__global__ // to perform kernel function --> must be void
void getmaxcu(unsigned int* numbers_all, unsigned int* result_all, int size){
  
  int i = 0;
  i = threadIdx.x + blockDim.x * blockIdx.x; // 1D grid of 1D blocks
  
  if (i < size){
    // reads the word old located in first param in global/shared memory, 
    // computes the max of old and value (second param), and stores the result 
    // back to memory at the same address (3 operations are performed in one
    // atomic transaction --> returns old)
    atomicMax((int*)result_all, (int)numbers_all[i]);
  }

}

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; // pointer to the array

    unsigned int * result;
    result = (unsigned int*)malloc(sizeof(unsigned int));
    result[0] = 0; // this is where the array max will be stored in
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for(i = 0; i < size; i++)
       numbers[i] = rand()  % size;    

    // this is where we're going to allocate and then copy over memory
    unsigned int * numbers_all;
    unsigned int * result_all;

    // transfer m and n to device memory
    cudaMalloc((void **)&numbers_all, size*sizeof(unsigned int));
    cudaMemcpy(numbers_all, numbers, size*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&result_all, sizeof(unsigned int));
    cudaMemcpy(result_all, result, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // getter and setter props
    int thread_num = 1024;
    int block_num = (int)ceil(size/(double)thread_num);

    getmaxcu<<<block_num, thread_num>>>(numbers_all, result_all, size); // calling the kernel here
    cudaMemcpy(result, result_all, sizeof(unsigned int), cudaMemcpyHostToDevice); // transfering from device to host
   
    // freeing the device memory!!!
    cudaFree(numbers_all);
    cudaFree(result_all);

    printf("The maximum number in the array is: %u\n", result[0]);

    free(numbers);
    exit(0);
}