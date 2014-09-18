// #CSCS CUDA Training 
//
// #Example 11 - time per thread operations
//
// #Author Ugo Varetto
//
// #Goal: time operations with per-multiprocessor counters
//
// #Rationale: shows how to use the clock() call wihin kernels
//
// #Solution: use the clock() function on the GPU to time operations 
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
// #Compilation: nvcc -arch=sm_13 11_clock.cu -o clock
//
// #Execution: ./clock 
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied       
//
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
 

//#include <cuda_runtime.h> // automatically added by nvcc
#include <iostream>

typedef double real_t;

// time clock invocation
__global__ void time_clock( clock_t* clkout ) {
    const clock_t start = clock();
    const clock_t stop  = clock();
    *clkout = stop - start;
}


__device__ real_t foo() { return 2.0; }

// time operation
__global__ void time_op( clock_t* clkout, real_t* out, int N, clock_t clock_offset ) {
   real_t f;
   clock_t acc = 0;
   __shared__ real_t local[ 1 ];
   for( int i = 0; i != N; ++i ) {
       const clock_t start = clock();
       local[ 0 ]= foo();
       const clock_t stop  = clock();
       acc += stop - start;
   } 
   *clkout = acc / N - clock_offset;
   *out = f;
}


//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    clock_t* dev_clk = 0;
    real_t* dev_out = 0;

    // allocate memory for clock and data output 
    cudaMalloc( &dev_clk, sizeof( clock_t ) );
    cudaMalloc( &dev_out, sizeof( real_t ) );
    
    // compute overhead of invoking clock()
    time_clock<<<1,1>>>( dev_clk );
    cudaThreadSynchronize();
    clock_t clk_offset = clock_t();
    cudaMemcpy( &clk_offset, dev_clk, sizeof( clock_t ), cudaMemcpyDeviceToHost );
    std::cout << "Clock overhead: " << clk_offset << std::endl;

    // time operation
    time_op<<<1,1>>>( dev_clk, dev_out, 100, clk_offset );
    cudaThreadSynchronize();
    clock_t clk = clock_t();
    cudaMemcpy( &clk, dev_clk, sizeof( clock_t ), cudaMemcpyDeviceToHost );
    
    // report
    std::cout << "Clock cycles:   " << clk << std::endl; 
    
    // free resources
    cudaFree( dev_clk );
    cudaFree( dev_out );

    return 0;
}
