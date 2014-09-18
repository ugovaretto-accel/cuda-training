// #CSCS CUDA Training 
//
// #Example 7 - constant memory
//
// #Author Ugo Varetto
//
// #Goal:  multiply input array elements by weights and write result into output array;
//         investigate the use of constant memory for storing the weights and how
//         having each thread in each warp access the same element compares
//         to have each thread access a different element from a performance standpoint;
//         also compare the result with storing data in global memory      
//
// #Rationale: constant memory can be used to store a small (64kiB) dataset frequently
//             accessed from a kernel; behavior of constant memory is the opposite of
//             global memory: it is faster to have multiple threads access the same
//             element than having multiple threads access separate elements
//
// #Solution: implement and time different kernels:
//            1) all the threads in a warp access the same element in the const array
//            2) each thread in the grid accesses a different element in the const array;
//            redo the same for global memory timing the computation with events
//
// #Code: flow:
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host to device, in this case also copy to const global array on GPU
//        4) launch and time kernels
//        6) synchronize events to wait for end of execution 
//        7) consume data (in this case print result and time)
//        8) free memory and events (used to time operations)
//        
// #Compilation: nvcc -arch=sm_13 7_constant-memory.cu -o constant-memory
//
// #Execution: ./constant-memory
//
// #Note: note how on arch 1.3 devices broadcast access is faster than parallel access in constant
//        memory (quite the opposite in global memory) because access to constant memory is serialized
//        within half-warp: each half-warp can only access one element of const memory at a time     
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
#include <sstream>
#include <algorithm>
#include <iterator>

typedef float real_t;

static const int HALF_WARP = 16;
static const int NUMBER_OF_BLOCKS  = 64;
static const int THREADS_PER_BLOCK = 16 * HALF_WARP;
static const int VECTOR_SIZE = THREADS_PER_BLOCK * NUMBER_OF_BLOCKS;
static const int NUM_WEIGHTS = VECTOR_SIZE;
static const int BYTE_SIZE = sizeof( real_t ) * VECTOR_SIZE;

// number of weights must fit in local memory = 64kiB
// 64 blocks x ( 16 x 16 threads per block ) = 16ki elements x 4 bytes per element = 64kiB   

__constant__ real_t weights[ NUM_WEIGHTS ];


// out[ global thread id ] = in[ global thread id ] x weights[ block offset + half warp id ];
// each thread in the half-warp accesses the same element in the constant weight array
__global__ void weight_mul_broadcast( const real_t* vin, real_t* out ) {
    // compute current thread id
    const int xBlock = blockIdx.x * blockDim.x;
    const int xIndex = xBlock + threadIdx.x;          
    out[ xIndex ] = vin[ xIndex ] * weights[ xBlock + threadIdx.x / HALF_WARP ];
}

// out[ global thread id ] = in[ global thread id ] x weights[ global thread id ];
// each thread  accesses a different weight: each access from half warp
// threads is serialized i.e. it will take 16 separate read operations to fill a group of 16 output
// elements as comparaed to 16 parallel transfers or less(when coalesced) in the case of global
// memory 
__global__ void weight_mul_separate( const real_t* vin, real_t* out ) {
    // compute current thread id
    const int xBlock = blockIdx.x * blockDim.x;
    const int xIndex = xBlock + threadIdx.x;              
    out[ xIndex ] = vin[ xIndex ] * weights[ xBlock + threadIdx.x ];
}


// same as weight_mul_parallel but reading weights from global memory
__global__ void weight_mul_global_separate( const real_t* vin, const real_t* w, real_t* out ) {
    // compute current thread id
    const int xBlock = blockIdx.x * blockDim.x;
    const int xIndex = xBlock + threadIdx.x;              
    out[ xIndex ] = vin[ xIndex ] * w[ xBlock + threadIdx.x ];
}

// same as weight_mul_broadcast but reading weights from global memory
__global__ void weight_mul_global_broadcast( const real_t* vin, const real_t* w, real_t* out ) {
    // compute current thread id
    const int xBlock = blockIdx.x * blockDim.x;
    const int xIndex = xBlock + threadIdx.x;              
    out[ xIndex ] = vin[ xIndex ] * w[  xBlock + threadIdx.x / HALF_WARP ];//+ threadIdx.x / HALF_WARP ];
}

// generate sequence {i,i+1,i+2,...}
struct GenSeq {
    static int v_; 
    GenSeq( real_t v )  { v_ = v; }
    real_t operator()() const { return v_++; }
};
int GenSeq::v_ = 0;


//------------------------------------------------------------------------------
int main( int , char**  ) {
         
    // host allocated storage; page locked memory required for async/stream operations
    std::vector< real_t > v( VECTOR_SIZE, 1.f );
    std::vector< real_t > host_w( NUM_WEIGHTS );
    std::vector< real_t > vout( VECTOR_SIZE );
   
    std::generate( host_w.begin(), host_w.end(), GenSeq( 0.0f ) );
    std::cout << "Input: ";
    std::copy( v.begin(), v.begin() + 10, std::ostream_iterator< real_t >( std::cout, ", ") ); 
    std::cout << " ..." << std::endl;
    std::cout << "Weigths: ";
    std::copy( host_w.begin(), host_w.begin() + 10, std::ostream_iterator< real_t >( std::cout, ", ") ); 
    std::cout << " ..." << std::endl;
 
    //upload data to const global on GPU (data are const on the GPU, must be initialized from the CPU)
    cudaMemcpyToSymbol( weights, &host_w[ 0 ], sizeof( real_t ) * NUM_WEIGHTS );
 
    // gpu allocated storage
    real_t* dev_vin  = 0;
    real_t* dev_vout = 0;
    real_t* dev_w    = 0;

    cudaMalloc( &dev_vin,  BYTE_SIZE );
    cudaMalloc( &dev_vout, BYTE_SIZE );
    cudaMalloc( &dev_w, sizeof( real_t) * NUM_WEIGHTS );

    cudaMemcpy( dev_vin, &v[ 0 ], BYTE_SIZE, cudaMemcpyHostToDevice );
    
    cudaMemcpy( dev_w,   &host_w[ 0 ], sizeof( real_t ) * NUM_WEIGHTS, cudaMemcpyHostToDevice );
   
    // events; for timing
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop  = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    float e = float();

    cudaEventRecord( start, 0 );
    weight_mul_broadcast<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_vout );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );    
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Const broadcast:  " << e << " ms" << std::endl;
    //copy data from GPU and print result
    cudaMemcpy( &vout[ 0 ], dev_vout, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Result: ";
    std::copy( vout.begin(), vout.begin() + 48, std::ostream_iterator< real_t >( std::cout, ", ") ); 
    std::cout << " ...\n" << std::endl;

    cudaEventRecord( start, 0 );
    weight_mul_separate<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_vout );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Const separate: " << e << " ms" << std::endl;
     //copy data from GPU and print result
    cudaMemcpy( &vout[ 0 ], dev_vout, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Result: ";
    std::copy( vout.begin(), vout.begin() + 48, std::ostream_iterator< real_t >( std::cout, ", ") ); 
    std::cout << " ...\n" << std::endl;

    cudaEventRecord( start, 0 );
    weight_mul_global_broadcast<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_w, dev_vout );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Global broadcast:     " << e << " ms" << std::endl;
    //copy data from GPU and print result
    cudaMemcpy( &vout[ 0 ], dev_vout, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Result: ";
    std::copy( vout.begin(), vout.begin() + 48, std::ostream_iterator< real_t >( std::cout, ", ") ); 
    std::cout << " ...\n" << std::endl;
  
    cudaEventRecord( start, 0 );
    weight_mul_global_separate<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_w, dev_vout );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Global separate:     " << e << " ms" << std::endl;
    //copy data from GPU and print result
    cudaMemcpy( &vout[ 0 ], dev_vout, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Result: ";
    std::copy( vout.begin(), vout.begin() + 48, std::ostream_iterator< real_t >( std::cout, ", ") ); 
    std::cout << " ...\n" << std::endl;

    // free memory
    cudaFree( dev_vin  );
    cudaFree( dev_w    );
    cudaFree( dev_vout );

    // release events
    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

    return 0;
}
