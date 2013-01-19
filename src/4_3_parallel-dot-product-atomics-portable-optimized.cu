// #CSCS CUDA Training 
//
// #Example 4.3 - dot product with two step reduction, all processing on GPU
//
// #Author: Ugo Varetto
//
// #Goal: compute the dot product of two vectors performing all the computation on the GPU 
//
// #Rationale: shows how to perform the dot product of two vectors as a parallel reduction
//             with all the computation on the GPU           
// 
// #Solution: use the same standard parallel reduction algorithm shown in examples 4.1 and 4.2 twice: 
//            1) each block produces stores a single result into an array
//            2) the last block to compute a partial reduction perform a parallel
//               reduction step on the array generated at (1) 
//            The first thread (0) of each block atomically increments a global counter then
//            checks the counter value: the last block to increment the counter is the one
//            that reads '(gridDim.x - 1)' from the counter.
//            Global synchronization is achieved through __threadfence() 
//            to ensure that all the elements in the output array have been written 

//
// #Code: 1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) initialize data directly on GPU
//        4) launch kernel
//        5) report errors 
//        6) read data back 
//        7) free memory 
//             
// #Compilation: 
//               nvcc -arch=sm_13 4_3_parallel-dot-product-atomics-portable-optimized.cu -o dot-product-atomics
//
// #Execution: ./dot-product-atomics
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied
//
// #Note: also check cudaMemset, cudaErrorString, cudaGetLastError usage
//
// #Note: as of CUDA 3.2 it seems that kernels do not stall anymore when invoking
//        __syncthreads from within an if block dependent on the thread id;
//       #see http://forums.nvidia.com/index.php?showtopic=178284
//


//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>


typedef float real_t;

const size_t BLOCK_SIZE = 16;

//------------------------------------------------------------------------------

//Full on-gpu reduction

// each block atomically increments this variable when done
// performing the first reduction step
__device__ unsigned int count = 0;
// shared memory used by partial_dot and sum functions
// for temporary partial reductions; declare as global variable
// because used in more than one function
__shared__ real_t cache[ BLOCK_SIZE ];

// partial dot product: each thread block produces a single value
__device__ real_t partial_dot( const real_t* v1, const real_t* v2, int N, real_t* out  ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= N ) return real_t( 0 );
    cache[ threadIdx.x ] = 0.f;
    // the threads in the thread block iterate over the entire domain; iteration happens
    // whenever the total number of threads is lower than the domain size
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }
    __syncthreads(); // required because later on the current thread is accessing
                     // data written by another thread    
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; 
    }
    return cache[ 0 ];
}

// sum all elements in array; array size assumed to be equal to number of blocks
__device__ real_t sum( const real_t* v ) {
    cache[ threadIdx.x ] = 0.f;
    int i = threadIdx.x;
    // the threads in the thread block iterate oevr the entire domain
    // of size == gridDim.x == total number of blocks; iteration happens
    // whenever the number of threads in a thread block is lower than
    // the total number of thread blocks
    while( i < gridDim.x ) {
        cache[ threadIdx.x ] += v[ i ];
        i += blockDim.x;
    }
    __syncthreads(); // required because later on the current thread is accessing
                     // data written by another thread        
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; 
    }
    return cache[ 0 ];
}

// perform parallel dot product in two steps:
// 1) each block computes a single value and stores it into an array of size == number of blocks
// 2) the last block to finish step (1) performs a reduction on the array produced in the above step
// parameters:
// v1 first input vector
// v2 second input vector
// N  size of input vector
// out output vector: size MUST be equal to the number of GPU blocks since it us used
//     for partial reduction; result is at position 0
__global__ void full_dot( const real_t* v1, const real_t* v2, int N, real_t* out ) {
    // true if last block to compute value
    __shared__ bool lastBlock;
    // each block computes a value
    real_t r = partial_dot( v1, v2, N, out );
    if( threadIdx.x == 0 ) {
        // value is stored into output array by first thread of each block
        out[ blockIdx.x ] = r;
        // wait for value to be available to all the threads on the device
        __threadfence();
        // increment atomic counter and retrieve value
        const unsigned int v = atomicInc( &count, gridDim.x );
        // check if last block to perform computation
        lastBlock = ( v == gridDim.x - 1 );
    }
    // the code below is executed by *all* threads in the block:
    // make sure all the threads in the block access the correct value
    // of the variable 'lastBlock'
    __syncthreads();
    
    // last block performs a the final reduction steps which produces one single value
    if( lastBlock ) {
        r = sum( out );
        if( threadIdx.x == 0 ) {
            out[ 0 ] =  r;
            count = 0;
        }
    }   
}

//------------------------------------------------------------------------------

// cpu implementation of dot product
real_t dot( const real_t* v1, const real_t* v2, int N ) {
    real_t s = 0;
    for( int i = 0; i != N; ++i ) {
        s += v1[ i ] * v2[ i ];
    }
    return s;
}

// initialization function run on the GPU
__global__ void init_vector( real_t* v, int N ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while( i < N ) {
        v[ i ] = 1.0f;//real_t( i ) / 1000000.f;
        i += gridDim.x * blockDim.x;
    } 
}


//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
    
    const size_t ARRAY_SIZE = 1024;//1024 * 1024; //1Mi elements
    const int BLOCKS = 64;//512;
    const int THREADS_PER_BLOCK = BLOCK_SIZE;//256; // total threads = 512 x 256 = 128ki threads;                                   
    const size_t SIZE = ARRAY_SIZE * sizeof( real_t );
    
    // device storage
    real_t* dev_v1  = 0; // vector 1
    real_t* dev_v2  = 0; // vector 2
    real_t* dev_out = 0; // result array, final result is at position 0;
                         // also used for temporary GPU storage,
                         // must have size == number of thread blocks  
    cudaMalloc( &dev_v1,  SIZE );
    cudaMalloc( &dev_v2,  SIZE );
    cudaMalloc( &dev_out, sizeof( real_t ) * BLOCKS );

    // host storage
    std::vector< real_t > host_v1( ARRAY_SIZE );
    std::vector< real_t > host_v2( ARRAY_SIZE );
    real_t host_out = 0.f;

    // initialize vector 1 with kernel; much faster than using for loops on the cpu
    init_vector<<< 1024, 256  >>>( dev_v1, ARRAY_SIZE );
    cudaMemcpy( &host_v1[ 0 ], dev_v1, SIZE, cudaMemcpyDeviceToHost );
    // initialize vector 2 with kernel; much faster than using for loops on the cpu
    init_vector<<< 1024, 256  >>>( dev_v2, ARRAY_SIZE );
    cudaMemcpy( &host_v2[ 0 ], dev_v2, SIZE, cudaMemcpyDeviceToHost );
      
    // execute kernel
    full_dot<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_v1, dev_v2, ARRAY_SIZE, dev_out );
    std::cout << cudaGetErrorString( cudaGetLastError() ) << std::endl;
             
    // copy output data from device(gpu) to host(cpu)
    cudaMemcpy( &host_out, dev_out, sizeof( real_t ), cudaMemcpyDeviceToHost );

    // print dot product by summing up the partially reduced vectors
    std::cout << "GPU: " << host_out << std::endl;    

    // print dot product on cpu
    std::cout << "CPU: " << dot( &host_v1[ 0 ], &host_v2[ 0 ], ARRAY_SIZE ) << std::endl;

    // free memory
    cudaFree( dev_v1 );
    cudaFree( dev_v2 );
    cudaFree( dev_out );

    return 0;
}
