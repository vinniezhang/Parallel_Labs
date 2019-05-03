// Vinnie Zhang
// Parallel Computing - Lab 3

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


// to perform kernel function (called from host code, executed on device) --> must be void
__global__ void getmaxcu(unsigned int* numbers_device, unsigned int* result_device, int array_size){ 
// numbers_device and result_device (first two params) point to device memory
  
  // 1D grid of 1D blocks of 1D threads --> threads form blocks form grid
  int i =   blockIdx.x * blockDim.x + threadIdx.x; 

  // blockDim.x used for threads per block
  
  if (i < array_size){ 
    // we don't want to exceed array size

    // reads the word old located in first param in global/shared memory, 
    // computes the max between the first and second param, and stores the result 
    // back to memory at the same address (first param) 
    // (3 operations are performed in one atomic transaction --> returns the max value, stored 
    // in first parameter)
    atomicMax((int*)result_device, (int)numbers_device[i]);
  }

}

// this is a less efficient way to retrieve max of array??
// __global__ void getmaxcu(unsigned int* numbers_device, unsigned int array_size)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     for (i = 0; i < size; ++i)
//     {
//         if (numbers_device[i] > numbers_device[0])
//             numbers_device[0] = numbers_device[i];
//     }
// }

int main(int argc, char *argv[])
{
    unsigned int array_size;  // size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; // pointer to the array
    
    unsigned int * result;
    final = (unsigned int*)malloc(sizeof(unsigned int)); // allocate space for host copies
    final[0] = 0; // this is the index where the max will be stored in --> is this correct?
    
    // given to us in sequential code file
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    array_size = atol(argv[1]); // converts string to a long int

    numbers = (unsigned int *)malloc(array_size * sizeof(unsigned int));
    if( !numbers ) {
       printf("Unable to allocate mem for an array of size %u\n", array_size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    
    // Fill-up the array with random numbers from 0 to size-1 
    for(i = 0; i < array_size; i++){
       numbers[i] = rand() % array_size; 
       // printf("%d", numbers[i]);
       // printf("\n");
    }

    // this is where the parallelizing comes in
    // we're going to allocate and then copy over memory
    unsigned int * numbers_device; 
    unsigned int * result_device;

    // transfer m and n to device memory and allocating space for device copies
    cudaMalloc((void **)&numbers_device, array_size*sizeof(unsigned int)); // allocating space for device copies in global memory
    cudaMemcpy(numbers_device, numbers, array_size*sizeof(unsigned int), cudaMemcpyHostToDevice); // copy input to device
    
    cudaMalloc((void **)&result_device, sizeof(unsigned int)); // allocating space for device copies in global memory
    cudaMemcpy(result_device, final, sizeof(unsigned int), cudaMemcpyHostToDevice); // copy result BACK to host

    // setting up input values
    int block_num = 32; // (int)ceil(array_size/(double)thread_num);
    int thread_num = 1024; // cims servers allow for this amount
    
    // call from host code to device code (aka kernal launch!!)
    getmaxcu<<<block_num, thread_num>>>(numbers_device, result_device, array_size);
    
    // this is where we copy the result back to host (from device) 
    cudaMemcpy(final, result_device, sizeof(unsigned int), cudaMemcpyDeviceToHost);
   
    // cleaning and freeing up the device memory!!!
    free(numbers);
    cudaFree(numbers_device);
    cudaFree(result_device);

    printf("The maximum number in the array is: %u\n", final[0]); // print statement, retrieving max value in array
    exit(0);

}