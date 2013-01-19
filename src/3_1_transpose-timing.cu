// #CSCS CUDA Training 
//
// #Example 3.1 - transpose matrix
//
// #Author: Ugo Varetto
//
// #Goal: compute the transpose of a matrix and time operation using
//        GPU's on-board performance counters through streams; print the result in ms (10^-3 s)    
//
// #Rationale: shows how to time GPU computation
//
// #Solution: straightworwad, simply compute the thread id associated with the element
//            and copy the transposed data into the output matrix; wrap kernel calls with event
//            recording and print time information         
//
// #Code: typical flow + timing: 
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) initialize data directly on the GPU
//        4) create events
//        5) record start time
//        6) launch kernel
//        7) synchronize events to guarantee that kernel execution is finished
//        8) record stop time
//        9) read data back 
//        10) print timing information as stop - start time 
//        11) delete events 
//        12) free memory      
//        The code uses the default stream 0; streams are used to sychronize operations
//        to guarantee that all operations in the same stream are executed sequentially.
//                
// #Compilation: nvcc -arch=sm_13 3_1_transpose-timing.cu -o transpose-timing 
//
// #Execution: ./transpose-timing
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
// #Note: the example can be extended to read configuration data and matrix size from the command line

//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>

typedef float real_t;

__global__ void transpose( const real_t* in, real_t *out, int num_rows, int num_columns ) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_index = row * num_columns + col;
    const int output_index = col * num_rows + row; 
    out[ output_index ] = in[ input_index ];
}

__global__ void init_matrix( real_t* in ) {
    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int r = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = c + gridDim.x * blockDim.x * r; 
    in[ idx ] = (real_t) idx; 
}

void print_matrix( const real_t* m, int r, int c, int stride ) {
    for( int i = 0; i != r; ++i ) {
        for( int j = 0; j != c; ++j ) std::cout << m[ i * stride + j ] << ' ';
        std::cout << '\n';
    }
    std::cout << std::endl;        
}

//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
    
    const dim3 BLOCKS( 512, 512 );
    const dim3 THREADS_PER_BLOCK( 16, 16 ); 
    const int ROWS = 512 * 16; // 8192
    const int COLUMNS =  512 * 16; // 8192
    const size_t SIZE = ROWS * COLUMNS * sizeof( real_t );
    
    // device storage
    real_t* dev_in = 0;
    real_t* dev_out = 0;
    cudaMalloc( &dev_in,  SIZE );
    cudaMalloc( &dev_out, SIZE );
    
    // host storage
    std::vector< real_t > outmatrix( ROWS * COLUMNS );

    // initialize matrix with kernel; much faster than using
    // for loops on the cpu
    init_matrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_in );
    cudaMemcpy( &outmatrix[ 0 ], dev_in, SIZE, cudaMemcpyDeviceToHost );

    // print upper 4x4 left corner of input matrix
    std::cout << "INPUT MATRIX - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
    print_matrix( &outmatrix[ 0 ], 4, 4, COLUMNS );
    
    // create events for timing execution
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop  = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    
    // record time into start event 
    cudaEventRecord( start, 0 ); // 0 is the default stream id
    // execute kernel
    transpose<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_in, dev_out, ROWS, COLUMNS );
    
    // issue request to record time into stop event
    cudaEventRecord( stop, 0 );
    // synchronize stop event to wait for end of kernel execution on stream 0
    cudaEventSynchronize( stop );
    // compute elapsed time (done by CUDA run-time) 
    float elapsed = 0.f;
    cudaEventElapsedTime( &elapsed, start, stop );
    
    std::cout << "Elapsed time (ms): " << elapsed << std::endl;

    // copy output data from device(gpu) to host(cpu)
    cudaMemcpy( &outmatrix[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    
    // print upper 4x4 corner of transposed matrix
    std::cout << "\nOUTPUT MATRIX - " << COLUMNS << " rows, " << ROWS << " columns" << std::endl;
    print_matrix( &outmatrix[ 0 ], 4, 4, ROWS );

    // free memory
    cudaFree( dev_in );
    cudaFree( dev_out );

    // release events
    cudaEventDestroy( start );
    cudaEventDestroy( stop  );


    return 0;
}
