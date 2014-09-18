// #CSCS CUDA Training 
//
// #Example 4 - dot product
//
// #Author: Ugo Varetto
//
// #Goal: compute the dot product of two vectors 
//
// #Rationale: shows how to perform the dot product of two vectors as a parallel reduction
//             with the latest reduction step performed on the CPU
// 
// #Solution: store scalar products in local cache and iterate over cache elements
//            performing incremental sums 
//
// #Code: 1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) initialize data directly on GPU
//        4) launch kernel
//        5) read data back 
//        6) perform last reduction step on CPU
//        7) free memory 
//             
// #Compilation: nvcc -arch=sm_13 4_parallel_dot_product.cu -o dot_product
//
// #Execution: ./dot_product
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied 
//
// #Note: as of CUDA 3.2 it seems that kernels do not stall anymore when invoking
//        __syncthreads from within an if block dependent on the thread id;
//       #see http://forums.nvidia.com/index.php?showtopic=178284
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision

//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <numeric>

typedef float real_t;

const size_t BLOCK_SIZE = 16;

__global__ void partial_dot( const real_t* v1, const real_t* v2, real_t* out, int N ) {
    __shared__ real_t cache[ BLOCK_SIZE ];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[ threadIdx.x ] = 0.f;
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
        i /= 2; //not sure bitwise operations are actually faster
    }

    if( threadIdx.x == 0 ) out[ blockIdx.x ] = cache[ 0 ];
}

real_t dot( const real_t* v1, const real_t* v2, int N ) {
    real_t s = 0;
    for( int i = 0; i != N; ++i ) {
        s += v1[ i ] * v2[ i ];
    }
    return s;
}


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
    const int THREADS_PER_BLOCK = BLOCK_SIZE;//256; // total threads = 512 x 256 = 128ki threads;                                   // each thread spans 8 array elements  
    const size_t SIZE = ARRAY_SIZE * sizeof( real_t );
    
    // device storage
    real_t* dev_v1 = 0; // vector 1
    real_t* dev_v2 = 0; // vector 2
    real_t* dev_vout = 0; // partial redution = number of blocks
    cudaMalloc( &dev_v1,  SIZE );
    cudaMalloc( &dev_v2,  SIZE );
    cudaMalloc( &dev_vout, BLOCKS * sizeof( real_t ) );

    // host storage
    std::vector< real_t > host_v1( ARRAY_SIZE );
    std::vector< real_t > host_v2( ARRAY_SIZE );
    std::vector< real_t > host_vout( BLOCKS );

    // initialize vector 1 with kernel; much faster than using for loops on the cpu
    init_vector<<< 1024, 256  >>>( dev_v1, ARRAY_SIZE );
    cudaMemcpy( &host_v1[ 0 ], dev_v1, SIZE, cudaMemcpyDeviceToHost );
    // initialize vector 2 with kernel; much faster than using for loops on the cpu
    init_vector<<< 1024, 256  >>>( dev_v2, ARRAY_SIZE );
    cudaMemcpy( &host_v2[ 0 ], dev_v2, SIZE, cudaMemcpyDeviceToHost );
       
    // execute kernel
    partial_dot<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_v1, dev_v2, dev_vout, ARRAY_SIZE );
         
    // copy output data from device(gpu) to host(cpu)
    cudaMemcpy( &host_vout[ 0 ], dev_vout, BLOCKS * sizeof( real_t ), cudaMemcpyDeviceToHost );

    // print dot product by summing up the partially reduced vectors
    std::cout << "GPU: " << std::accumulate( host_vout.begin(), host_vout.end(), real_t( 0 ) ) << std::endl;    

    // print dot product on cpu
    std::cout << "CPU: " << dot( &host_v1[ 0 ], &host_v2[ 0 ], ARRAY_SIZE ) << std::endl;

    // free memory
    cudaFree( dev_v1 );
    cudaFree( dev_v2 );
    cudaFree( dev_vout );

    return 0;
}
