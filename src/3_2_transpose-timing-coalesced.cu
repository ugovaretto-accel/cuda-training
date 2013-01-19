// #CSCS CUDA Training 
//
// #Example 3.2 - transpose matrix, coalesced access
//
// #Author: Ugo Varetto
//
// #Goal: compute the transpose of a matrix with coalesced memory access 
//
// #Rationale: shows how to increase speed by making use of shared (among threads in a thread block) memory
//             and coalesced memory access.
//             CUDA can perform a memory transfer of 16(half warp) contiguous elements(4,8 or 16 bytes each)
//             in a single step if each thread accesses a different memory location in the 16 element buffer;
//             also shared memory access can be two orders of magnitude faster than global memory access
//               
//
// #Solution: copy input matrix elements into shared memory blocks and write transposed elements
//            reading from shared memory. Access is coalesced if the block size is a multiple
//            of a half warp i.e. 16
//
// #Code: 1) compute launch grid configuration
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
// #Compilation: nvcc -arch=sm_13 3_2_transpose-timing-coalesced.cu -o transpose-timining-coalesced
//
// #Execution: ./transpose-timining-coalesced
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
//
// #Note: despite improvements in Tesla2 and Fermi hardware, coalescing is BY NO MEANS obsolete.
//        Even on Tesla2 or Fermi class hardware, failing to coalesce global memory transactions
//        can result in a 2x performance hit. (On Fermi class hardware, this seems to be true only
//        when ECC is enabled. Contiguous-but-uncoalesced memory transactions take about a 20% hit on Fermi.)

//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>

typedef float real_t;

const size_t TILE_SIZE = 16; //16 == half warp -> coalesced access

__global__ void transpose( const real_t* in, real_t *out, int num_rows, int num_columns ) {
    // local cache
    __shared__ real_t tile[ TILE_SIZE ][ TILE_SIZE ];
    // locate element to transfer from input data into local cache
    // CAVEAT: size of tile == size of thread block i.e. blockDim.x == blockDim.y == TILE_SIZE
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int input_index = row * num_columns + col;
    // 1) copy data into tile
    tile[ threadIdx.y ][ threadIdx.x ] = in[ input_index ];
    // wait for all threads to perform copy operation since the threads that
    // write data to the output matrix must read data which has been written into cache
    // by different threads
    // 2) locate output element of transposed matrix
    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    // transposed matrix: num_columns -> num_rows == matrix width
    const int output_index = row * num_rows + col;
    // read data of transposed element from tile    
    __syncthreads();
    out[ output_index ] = tile[ threadIdx.x ][ threadIdx.y ];
    // note that (1) and (2) are completely separate and independent step
    // the only requirement for (2) to work is that the data are 
    // available in shared memory 
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
    //transposeCoalesced<<<BLOCKS, THREADS_PER_BLOCK>>>>( dev_in, dev_out, COLUMNS, ROWS);
    // issue request to record time into stop event
    cudaEventRecord( stop, 0 );
    // synchronize stop event to wait for end of kernel execution on stream 0
    cudaEventSynchronize( stop );
    // compute elapsed time (done by CUDA run-time) 
    float elapsed = 0.f;
    cudaEventElapsedTime( &elapsed, start, stop );
    
    std::cout << "Elapsed time (ms): " << elapsed  << std::endl;

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
