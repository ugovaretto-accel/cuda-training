// #CSCS CUDA Training 
//
// #Example 8 - texture memory
//
// #Author Ugo Varetto
//
// #Goal: compute the sum of two 1D vectors, compare performance of texture vs global
//        memory for storing the arrays
//
// #Rationale: shows how to use texture memory and that texture memory is not faster
//             in cases where input data are not re-used
//
// #Solution: same as ex. 2; add kernel which reads input data from texture memory and
//            properly initialize, map and release texture memory in driver code
//
// #Code: 1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) map texture memory to pre-allocated gpu storage
//        4) copy data from host to device
//        5) launch kernel
//        6) read data back
//        7) consume data (in this case print result)
//        8) release texture memory 
//        9) free memory
//        
// #Compilation: nvcc -arch=sm_13 8_0_texture-memory.cu -o texture-memory-1
//
// #Execution: ./texture-memory-1 
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision


//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

typedef float real_t;

// per-traslation unit (i.e. source file with all the includes) global variables
// to be accessed from within gpu code
texture< real_t > v1Tex;
texture< real_t > v2Tex;

// read input data from global memory
__global__ void sum_vectors( const real_t* v1, const real_t* v2, real_t* out, size_t num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do note write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = v1[ xIndex ] + v2[ xIndex ];
}

// read input data from texture memory
__global__ void sum_vectors_texture( real_t* out, size_t num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do note write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = tex1Dfetch( v1Tex, xIndex ) + tex1Dfetch( v2Tex, xIndex );
}


//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int VECTOR_SIZE = 0x10000 + 1; //vector size 65537
    const int SIZE = sizeof( real_t ) * VECTOR_SIZE; // total size in bytes
    const int THREADS_PER_BLOCK = 32; //number of gpu threads per block
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int NUMBER_OF_BLOCKS = ( VECTOR_SIZE + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    // if number of threads is not evenly divisable by the number of threads per block 
    // we need an additional block; the above code can be rewritten as
    // if( NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK;
    // else BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1 

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

    // describe data inside texture: 1-component floating point value in this case    
    const int BITS_PER_BYTE = 8;
    cudaChannelFormatDesc cd = cudaCreateChannelDesc( sizeof( real_t ) *  BITS_PER_BYTE,
                                                      0, 0, 0, cudaChannelFormatKindFloat );
    // bind textures to pre-allocated storage
    cudaBindTexture( 0, &v1Tex, dev_in1, &cd, SIZE );
    cudaBindTexture( 0, &v2Tex, dev_in2, &cd, SIZE );

    // copy data to GPU
    cudaMemcpy( dev_in1, &v1[ 0 ], SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_in2, &v2[ 0 ], SIZE, cudaMemcpyHostToDevice );

    // initialize events for timing execution
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    float e = 0.f;

    cudaEventRecord( start );
    
    // execute kernel accessing global memory
    sum_vectors<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>( dev_in1, dev_in2, dev_out, VECTOR_SIZE );
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    // print first and last element of vector
    std::cout << "Result: " << vout[ 0 ] << ".." << vout.back() << std::endl;
    std::cout << "Global memory:  " << e << " ms" << std::endl; 

    cudaEventRecord( start );
    // execute kernel accessing texture memory; input vectors are read from texture references
    sum_vectors_texture<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>( dev_out, VECTOR_SIZE );
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    // print first and last element of vector
    std::cout << "Result: " << vout[ 0 ] << ".." << vout.back() << std::endl;
    std::cout << "Texture memory: " << e << " ms" << std::endl; 

    // release (un-bind/un-map) textures
    cudaUnbindTexture( &v1Tex );
    cudaUnbindTexture( &v2Tex );

    // free memory
    cudaFree( dev_in1 );
    cudaFree( dev_in2 );
    cudaFree( dev_out );

    return 0;
}
