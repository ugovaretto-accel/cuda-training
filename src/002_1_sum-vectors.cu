// #CSCS CUDA Training 
//
// #Example 2.1 - sum vectors, fix number of threads
//
// #Author Ugo Varetto
//
// #Goal: compute the scalar product of two 1D vectors using a number of threads lower than the 
//        size of the output vector.
//
// #Rationale: shows how to implement a kernel with a computation/memory configuration independent on the 
//             domain data layout; this is required in case the data is bigger than the computation grid (see exercise 1)
//
// #Solution: 
//          Given the maximum number of threads to use compute number of blocks 
//          . total number of threads = T
//          . number of threads per block = Tb          
//          The number of blocks is = Tb div T; note that it doesn't matter if the integer division
//          gives a reminder since each GPU thread will iterate over multiple elements and ensure that
//          the entire domain is processed regardless of the number of threads concurrently executed
//
// #Code: typical flow:
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host to device
//        4) launch kernel
//        5) read data back
//        6) consume data (in this case print result)
//        7) free memory
//        
// #Compilation: nvcc -arch=sm_13 2_1_sum-vectors.cu -o sum-vectors-2
//
// #Execution: ./sum-vectors-2 
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied   
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
//
// #Note: the example can be extended to read configuration data and array size from the command line
//        and could be timed to investigate how performance is dependent on single/double precision
//        and thread block size


//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

typedef float real_t;

// In this case the number of GPU threads is smaller than the number of elements in the domain:
//  every iterates over multple elements to ensure than the entire domain is covered
__global__ void sum_vectors( const real_t* v1, const real_t* v2, real_t* out, size_t num_elements ) {
    // compute current thread id
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // iterate over vector: grid can be smaller than vector, it is therefore
    // required that each thread iterate over more than one vector element
    while( xIndex < num_elements ) {
        out[ xIndex ] = v1[ xIndex ] + v2[ xIndex ];
        xIndex += gridDim.x * blockDim.x;
    }
}


//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int VECTOR_SIZE = 0x10000 + 1; //vector size 65537
    const int MAX_NUMBER_OF_THREADS = VECTOR_SIZE / 5;
    const int SIZE = sizeof( real_t ) * VECTOR_SIZE; // total size in bytes
    const int THREADS_PER_BLOCK = 32; //number of gpu threads per block
    const int NUMBER_OF_BLOCKS = MAX_NUMBER_OF_THREADS / THREADS_PER_BLOCK; 
   
    // host allocated storage; use std vectors to simplify memory management
    // and initialization
    std::vector< real_t > v1  ( VECTOR_SIZE, 1.f ); //initialize all elements to 1
    std::vector< real_t > v2  ( VECTOR_SIZE, 2.f ); //initialize all elements to 2   
    std::vector< real_t > vout( VECTOR_SIZE, 0.f ); //initialize all elements to 0

    // gpu allocated storage
    real_t* dev_in1 = 0; //vector 1
    real_t* dev_in2 = 0; //vector 2
    real_t* dev_out = 0; //result value
    cudaMalloc( &dev_in1, SIZE );
    cudaMalloc( &dev_in2, SIZE );
    cudaMalloc( &dev_out, SIZE  );
    
    // copy data to GPU
    cudaMemcpy( dev_in1, &v1[ 0 ], SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_in2, &v2[ 0 ], SIZE, cudaMemcpyHostToDevice );

    // execute kernel with num threads >= num elements
    sum_vectors<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>( dev_in1, dev_in2, dev_out, VECTOR_SIZE );
    
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    
    // print first and last element of vector
    std::cout << "result: " << vout.front() << ".." << vout.back() << std::endl;

    // free memory
    cudaFree( dev_in1 );
    cudaFree( dev_in2 );
    cudaFree( dev_out );

    return 0;
}
