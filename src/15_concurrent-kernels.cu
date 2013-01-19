// #CSCS CUDA Training 
//
// #Example 15 - concurrent kernels, requires Fermi cards for parallel execution
//
// #Author Ugo Varetto
//
// #Goal: execute kernels on the GPU in parallel, check serialized versus parallel performance
//    
// #Rationale: Fermi-based cards allow kernels to run in parallel
//
// #Solution: 
//           . run each kernel in a separate stream
//           . record a global start and stop event before and after execution of the kernel
//             launch loop
//           . sync stop event
//           . [optional] have each kernel store its timing information into an array
//             and use a separate kernel to report timing information after all other
//             kernel have been executed - requires stream to wait for the recording
//             of events in the other streams
// 
// #Code: flow:
//        1) create one stream per kernel + one stream for the time reporting kernel
//        2) create one sync event per kernel: the time reporting kernel will wait
//           for all the event to be recoreded before it gets executed
//        3) create start/stop events in the global scope i.e. not associated with any stream 
//        4) allocate data on host(cpu) and device(gpu), async memory transfers do require
//           page-locked memory
//        5) record start event
//        6) launch asynchronously kernels in a loop; support both independent(parallel)
//           and dependent(serial) execution:
//           - parallel: have the time reporting stream wait for each kernel to finish execution,
//            this is achieved by recording an event after each kernel launch and making 
//             the stream wait on the event through a call to cudaStreamWaitEvent
//           - serial: same as above but always use the same(first) stream 
//        7) launch asynchronously the time reporting kernel
//        8) issue ansynchronous mem-copy operations
//        9) sync everything, synchronizing with a global(i.e. not associated with any stream)
//           event is enough because it requires all the pending operations in all streams
//           to be completed
//       10) output information
//       11) free memory, streams and events 
//        
// #Compilation: [concurrent execution(on Fermi)] nvcc 15_concurrent-kernels.cu -o concurrent-kernels
//               [serial execution] nvcc 15_concurrent-kernels.cu -o concurrent-kernels -DFORCE_SERIALIZED
//                 
// #Execution: ./concurrent-kernels
//
// #Note: 2.x architecture required for parallel kernel execution
//
// #Note: the maximum number of concurrent kernels on 2.0 architecture with CUDA SDK 4.0
//        seems to be 16
//              
// #Note: page locked memory required for async/stream operations

//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <algorithm>
#include <ctime>

//sleep for the requested number of clocks
__global__ void timed_kernel( clock_t* clocksArray, int kernelIdx, int clockTicks ) {
    const clock_t start = clock();
    clock_t elapsed = 0;
    while( elapsed < clockTicks ) elapsed = clock() - start; 
    clocksArray[ kernelIdx ] = elapsed;
}

//parallel reduction: assume only one thread block used for computation;
//using more than a single block requires inter-block sychronization, see example 4.1/4.2 
__global__ void sum_clocks( clock_t* result, const clock_t* clocks, int numElements ) {
    const int CACHE_SIZE = 32; // equal to number of threads in thread block
    __shared__ clock_t cache[ CACHE_SIZE ];
    cache[ threadIdx.x ] = 0;
    if( threadIdx.x < numElements ) {
        for( int i = 0; i < numElements; i += CACHE_SIZE ) {
            cache[ threadIdx.x ] += clocks[ threadIdx.x + i ];   
        }
    }
    __syncthreads();
    for( int i = CACHE_SIZE / 2; i > 0; i /= 2 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
    }        
    result[ 0 ] = cache[ 0 ];
}

//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    //first task: verify support for concurrent kernel execution
    cudaDeviceProp prop = cudaDeviceProp();
    int currentDevice = -1;
    cudaGetDevice( &currentDevice );
    cudaGetDeviceProperties( &prop, currentDevice );
    if( prop.concurrentKernels == 0 ) {
        std::cout << "Concurrent kernel execution not supported\n"
                  << "kernels will be serialized" << std::endl;
    }    

    // change this value to find the maximum number of concurrent kernels supported
    const int NUM_KERNELS = 8;
    const int NUM_CLOCKS  = NUM_KERNELS;
    const size_t CLOCKS_BYTE_SIZE = NUM_CLOCKS * sizeof( clock_t );
    const int KERNEL_EXECUTION_TIME_ms = 50; 
    float elapsed_time = 0.f;   
    cudaEvent_t start, stop;
    std::vector< cudaEvent_t >  kernel_events( NUM_KERNELS );
    cudaStream_t time_compute_stream;
    std::vector< cudaStream_t > kernel_streams( NUM_KERNELS );

    //create timing events
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );

    //create kernel events
    for( std::vector< cudaEvent_t >::iterator i =  kernel_events.begin();
         i != kernel_events.end(); ++i ) {
        cudaEventCreateWithFlags( &(*i), cudaEventDisableTiming );             
        
    }

    //create stream for time reporting kernel: stream must wait for all kernel events to be recorded 
    cudaStreamCreate( &time_compute_stream );
    
    //create kernel streams
    for( std::vector< cudaStream_t >::iterator i =  kernel_streams.begin();
        i != kernel_streams.end(); ++i ) {
        cudaStreamCreate( &(*i) );           
    }

    //data array to hold timing information from kernel runs; TODO: use std::vector with page locked allocator
    clock_t* clocks    = 0;
    clock_t* clock_sum = 0; // sum of kernel execution times
    //we need host-allocated page locked memory because later-on an async memcpy operation is
    //is used; async operations *always* require page-locked memory
    cudaHostAlloc( &clocks, CLOCKS_BYTE_SIZE, cudaHostAllocPortable );
    cudaHostAlloc( &clock_sum, sizeof( clock_t ), cudaHostAllocPortable );
    clock_t* dev_clocks = 0;
    cudaMalloc( &dev_clocks, CLOCKS_BYTE_SIZE );
    clock_t* dev_clock_sum = 0;
    cudaMalloc( &dev_clock_sum, sizeof( clock_t ) );

    const int CLOCK_FREQ_kHz = prop.clockRate; 
    // BEGIN of async operations
    cudaEventRecord( start, 0 );
    clock_t cpu_start = clock();
    for( int k = 0; k != NUM_KERNELS; ++k ) {
#ifdef FORCE_SERIALIZED
        // clock ticks = freq [s^-1] x time [s]  =
        //   10 ^ 3 x freq  x 10 ^ -3 time       =
        //   CLOCK_FREQ_kHz x KERNEL_EXECUTION_TIME_ms
        timed_kernel<<< 1, 1, 0, kernel_streams[ 0 ] >>>( dev_clocks,
                                                          k,
                                                          CLOCK_FREQ_kHz * KERNEL_EXECUTION_TIME_ms );
        if( k == NUM_KERNELS - 1 ) { // record event after all kernel have been executed
             cudaEventRecord( kernel_events[ 0 ], kernel_streams[ 0 ] );
             cudaStreamWaitEvent( time_compute_stream, kernel_events[ 0 ], 0 /*must be zero*/ );
        }
#else
        timed_kernel<<< 1, 1, 0, kernel_streams[ k ] >>>( dev_clocks,
                                                          k,
                                                          KERNEL_EXECUTION_TIME_ms * CLOCK_FREQ_kHz );
        cudaEventRecord( kernel_events[ k ], kernel_streams[ k ] );
        cudaStreamWaitEvent( time_compute_stream, kernel_events[ k ], 0 /*must be zero*/ );
#endif               
    }
    const int NUM_BLOCKS = 1;
    const int NUM_THREADS_PER_BLOCK = 32; // must match shared memory size
    const size_t SHARED_MEMORY_SIZE = 0;     
    
    sum_clocks<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK,
                  SHARED_MEMORY_SIZE, time_compute_stream >>>( dev_clock_sum, dev_clocks, NUM_KERNELS );
    cudaMemcpyAsync( clock_sum, dev_clock_sum, sizeof( clock_t ), cudaMemcpyDeviceToHost, time_compute_stream );
    cudaMemcpyAsync( clocks, dev_clocks, CLOCKS_BYTE_SIZE, cudaMemcpyDeviceToHost, time_compute_stream );
    
    //record event, not associated with any stream and therefore recorded
    //after *all* stream events are recorded
    cudaEventRecord( stop, 0 );
    // END of async operations
    
    //sync everything
    //this synchronization call forces to wait until the stop event has been recorded;
    //the stop event is associated with the global context (the '0' in the cudaEventRegister call)
    //and therefore all events in the context must have been recorded before the stop event is recorded
    cudaEventSynchronize( stop );
    const double cpu_elapsed_time = clock() - cpu_start;
    cudaEventElapsedTime( &elapsed_time, start, stop );    
 
    //output information
    std::cout << "Clock:                                 " << double( CLOCK_FREQ_kHz ) * 1E-6 << " GHz" << std::endl; 
    std::cout << "Number of kernels:                     " << NUM_KERNELS << std::endl;
    std::cout << "Requested kernel execution time:       " << KERNEL_EXECUTION_TIME_ms << " ms" << std::endl;
    std::cout << "Computed kernel execution time:        " 
              << double( *std::max_element( clocks, clocks + NUM_KERNELS ) ) / CLOCK_FREQ_kHz << " ms" << std::endl;  
    std::cout << "Sum of kernel execution times:         " << double( *clock_sum ) / CLOCK_FREQ_kHz << " ms" << std::endl;  
    std::cout << "Total measured execution time:         " << elapsed_time << " ms" << std::endl;
    std::cout << "CPU elapsed time:                      " << 1000. * cpu_elapsed_time / CLOCKS_PER_SEC << " ms" << std::endl;
    //free resources
    for( std::vector< cudaEvent_t >::iterator i =  kernel_events.begin();
         i != kernel_events.end(); ++i ) {
        cudaEventDestroy( *i );            
    }

    //create sync stream: sync stream wait for all kernel events to be recorded 
    cudaStreamDestroy( time_compute_stream );
    
    //create kernel streams
    for( std::vector< cudaStream_t >::iterator i =  kernel_streams.begin();
         i != kernel_streams.end(); ++i ) {
        cudaStreamDestroy( *i );           
    }

    cudaFreeHost( clock_sum );
    cudaFreeHost( clocks );
    cudaFree( dev_clocks );
    cudaFree( dev_clock_sum );

    //OPTIONAL, apparently it must be called in order for profiling and tracing tools
    //to show complete traces
    cudaDeviceReset(); 

    return 0;

}
