// #CSCS CUDA Training 
//
// #Example 2.2 - sum vectors, fix number of threads per block, overlap communication and computation
//
// #Author Ugo Varetto
//
// #Goal: compute the sum of two vectors overlapping communication and computation
//
// #Rationale: using streams it is possible to subdivide computation and memory transfer
//             operations in separate execution queues which can be executed in parallel;
//             specifically it is possible to execute kernel computation while concurrently
//             transferring data between host and device
//
// #Solution: subdivide domain into chunks, iterate over the array and at each
//            iteration issue asynchronous calls to memcpy and kernels
//            (always asynchronous) in separate streams. Note that
//            on the GPU operations are split into one queue per operation type
//            specifically: all copy operations from different streams go into the same
//            queue in the same order as specified by client code, likewise all
//            kernel invocations go into the same 'kernel invocation queue'.
//            Now: if any copy operation (regardless of the associated stream) depends
//            on e.g. a kernel execution then all subsequent copy operations in all streams must wait
//            for the dependent copy operation to complete before they are executed; this means
//            that client code is responsible for properly queueing operations to avoid conflicts. 
//
// #Code: flow:
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host ro device
//        4) create streams
//        5) iterate over array elements performing memory transfers and
//           kernel invocation: at each iteration the number of elements
//           being processed by separate streams is VECTOR_CHUNK_SIZE x NUMBER_OF_STREAMS
//        6) synchronize streams to wait for end of execution 
//        7) consume data (in this case print result)
//        8) free memory, streams and events (used to time operations)
//        
// #Compilation: [optimized] nvcc -arch=sm_13 2_2_sum-vectors-overlap.cu -o sum_vectors-overlap
//               [wrong ordering, no overlap] nvcc -DSTREAM_NO_OVERLAP -arch=sm_13 2_2_sum-vectors-overlap.cu -o sum_vectors-overlap
//                 
//
// #Execution: ./sum-vectors-overlap
//             note that you might experience some hysteresis! on Win 7 64bit, CUDA RC2 at each compilation
//             it takes a few runs before the new ordering scheme becomes active!!! 
//
// #Note: page locked memory required for async/stream operations
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
// #Note: the example can be extended to read configuration data and array size from the command line and
//        investigating the optimal configuration for number of streams and chunk size


//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>

typedef float real_t;

// In this case the kernel assumes that the computation was started with enough threads to cover the entire domain.
// This is the preferred solution provided there are enough threads to cover the entire domain which might not be the
// case in case of a 1D grid layout (max number of threads = 512 threads per block x 65536  blocks = 2^25 = 32 Mi threads)
__global__ void sum_vectors( const real_t* v1, const real_t* v2, real_t* out, size_t num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do not write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = v1[ xIndex ] + v2[ xIndex ];
}


//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    //first task: verify support for overlap of communication and computation
    cudaDeviceProp prop = cudaDeviceProp();
    int currentDevice = -1;
    cudaGetDevice( &currentDevice );
    cudaGetDeviceProperties( &prop, currentDevice );
    if( !prop.deviceOverlap ) {
        std::cout << "Device doesn't handle computation-communication overlap" << std::endl;
        return 1;
    }

    const int VECTOR_SIZE = 0x1000000; //vector size 16Mi
    const int NUMBER_OF_CHUNKS = 4;
    const int VECTOR_CHUNK_SIZE = VECTOR_SIZE / NUMBER_OF_CHUNKS;
    const int FULL_BYTE_SIZE = sizeof( real_t ) * VECTOR_SIZE;
    const int CHUNK_BYTE_SIZE = FULL_BYTE_SIZE / NUMBER_OF_CHUNKS; // total size in bytes
    const int THREADS_PER_BLOCK = 256; //number of gpu threads per block
    const int NUMBER_OF_STREAMS = 2;

    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int NUMBER_OF_BLOCKS = ( VECTOR_CHUNK_SIZE + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    // if number of threads is not evenly divisable by the number of threads per block 
    // we need an additional block; the above code can be rewritten as
    // if( NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK;
    // else BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1 

    // host allocated storage; page locked memory required for async/stream operations
    real_t* v1   = 0;
    real_t* v2   = 0;
    real_t* vout = 0;
    
    // page locked allocation 
    cudaHostAlloc( &v1,   FULL_BYTE_SIZE, cudaHostAllocDefault );
    cudaHostAlloc( &v2,   FULL_BYTE_SIZE, cudaHostAllocDefault );
    cudaHostAlloc( &vout, FULL_BYTE_SIZE, cudaHostAllocDefault );  

    // generate constant element
    struct Gen {
        real_t v_;
        Gen( real_t v ) : v_( v ) {} 
        real_t operator()() const { return v_; }
    };

    std::generate( v1,   v1   + VECTOR_SIZE, Gen( 1.0f ) );
    std::generate( v2,   v2   + VECTOR_SIZE, Gen( 2.0f ) );
    std::generate( vout, vout + VECTOR_SIZE, Gen(  0.f ) );

    // gpu allocated storage: number of arrays == number of streams == 2
    real_t* dev_in00 = 0; //v1
    real_t* dev_in01 = 0; //v1
    real_t* dev_in10 = 0; //v2
    real_t* dev_in11 = 0; //v2
    real_t* dev_out0 = 0; //vout
    real_t* dev_out1 = 0; //vout

    cudaMalloc( &dev_in00, CHUNK_BYTE_SIZE );
    cudaMalloc( &dev_in01, CHUNK_BYTE_SIZE );
    cudaMalloc( &dev_in10, CHUNK_BYTE_SIZE );
    cudaMalloc( &dev_in11, CHUNK_BYTE_SIZE );
    cudaMalloc( &dev_out0, CHUNK_BYTE_SIZE );
    cudaMalloc( &dev_out1, CHUNK_BYTE_SIZE );
    
    // streams; each streams is associated with a separate execution queue
    cudaStream_t stream0 = cudaStream_t();
    cudaStream_t stream1 = cudaStream_t();
    cudaStreamCreate( &stream0 );
    cudaStreamCreate( &stream1 );

    // events; for timing
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop  = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    //record start
    cudaEventRecord( start, 0 );

#if defined( STREAM_NO_OVERLAP )
    // computation (wrong order): 
    for( int i = 0; i < VECTOR_SIZE; i += NUMBER_OF_STREAMS * VECTOR_CHUNK_SIZE ) {
        cudaMemcpyAsync( dev_in00, v1 + i, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream0 );
        cudaMemcpyAsync( dev_in10, v2 + i, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream0 );
        sum_vectors<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream0 >>>( dev_in00, dev_in10, dev_out0, VECTOR_CHUNK_SIZE );
        cudaMemcpyAsync( vout + i, dev_out0, CHUNK_BYTE_SIZE, cudaMemcpyDeviceToHost, stream0 );

        cudaMemcpyAsync( dev_in01, v1 + i + VECTOR_CHUNK_SIZE, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream1 );
        cudaMemcpyAsync( dev_in11, v2 + i + VECTOR_CHUNK_SIZE, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream1 );
        sum_vectors<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream1 >>>( dev_in01, dev_in11, dev_out1, VECTOR_CHUNK_SIZE );
        cudaMemcpyAsync( vout + i + VECTOR_CHUNK_SIZE, dev_out1, CHUNK_BYTE_SIZE, cudaMemcpyDeviceToHost, stream1 );
    }
#else
    // computation (correct order, interleaved or not makes little difference)
    for( int i = 0; i < VECTOR_SIZE; i += NUMBER_OF_STREAMS * VECTOR_CHUNK_SIZE ) {
        cudaMemcpyAsync( dev_in00, v1 + i, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream0 );
        cudaMemcpyAsync( dev_in01, v1 + i + VECTOR_CHUNK_SIZE, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream1 );
        cudaMemcpyAsync( dev_in10, v2 + i, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream0 );
        cudaMemcpyAsync( dev_in11, v2 + i + VECTOR_CHUNK_SIZE, CHUNK_BYTE_SIZE, cudaMemcpyHostToDevice, stream1 );
        sum_vectors<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream0 >>>( dev_in00, dev_in10, dev_out0, VECTOR_CHUNK_SIZE );  
        sum_vectors<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream1 >>>( dev_in01, dev_in11, dev_out1, VECTOR_CHUNK_SIZE );
        cudaMemcpyAsync( vout + i, dev_out0, CHUNK_BYTE_SIZE, cudaMemcpyDeviceToHost, stream0 );
        cudaMemcpyAsync( vout + i + VECTOR_CHUNK_SIZE, dev_out1, CHUNK_BYTE_SIZE, cudaMemcpyDeviceToHost, stream1 );
    }
#endif
    
    cudaEventRecord( stop, 0 );
    cudaStreamSynchronize( stream0 );
    cudaStreamSynchronize( stream1 );
    cudaEventSynchronize( stop );
    float e = float();
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "elapsed time (ms): " << e << std::endl;   
    // print first and last element of vector
    std::cout << "result: " << vout.front() << ".." << vout[ VECTOR_SIZE - 1 ] << std::endl;
     

    // free memory
    cudaFree( dev_in00 );
    cudaFree( dev_in01 );
    cudaFree( dev_in10 );
    cudaFree( dev_in11 );
    cudaFree( dev_out0 );
    cudaFree( dev_out1 );
    cudaFreeHost( v1 );
    cudaFreeHost( v2 );
    cudaFreeHost( vout );
    // release streams
    cudaStreamDestroy( stream0 );
    cudaStreamDestroy( stream1 ); 
    // release events
    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

    return 0;
}
