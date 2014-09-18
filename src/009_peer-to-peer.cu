// #CSCS CUDA Training 
//
// #Example 9 - CUDA 4, peer to peer access
//
// #Author Ugo Varetto
//
// #Goal: run kernels on separate GPUs passing the same pointer to both kernels
//
// #Rationale: shows how the same memory buffer can be accessed from kernels on separate GPUs 
//
// #Solution: use setCudaDevice and cudaEnablePeerAccess to select device and
//            enable sharing of memory
//
// #Code: 1) allocate device memory
//        2) select first GPU
//        3) launch kernel
//        4) copy data back from GPU 
//        5) select second GPU
//        6) launch other kernel
//        7) copy data back from GPU 
//        8) free memory
//        
// #Compilation: nvcc -arch=sm_20 9_peer-to-peer.cu -o peer-to-peer
//
// #Execution: ./peer-to-peer
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied
//
// #Note: Fermi (2.0) or better required; must be compiled with -arch=sm_2x
//
// #Note: Requires at least two GPUs



//#include <cuda_runtime.h> // automatically added by nvcc
#include <iostream>
#include <vector>

typedef float real_t;

__global__ void kernel_on_dev1( real_t* buffer ) {
    buffer[ blockIdx.x ] = 3.0;  
}

__global__ void kernel_on_dev2( real_t* buffer ) {
    buffer[ blockIdx.x ] *= 2.0;  
}

//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    real_t* dev_buffer = 0;
    const size_t SIZE = 1024;
    const size_t BYTE_SIZE = SIZE * sizeof( real_t );
    int ndev = 0;
    cudaGetDeviceCount( &ndev );
    if( ndev < 2 ) {
        std::cout << "At least two GPU devices required, " << ndev << " found" << std::endl;
    }
    
    // on device 0
    cudaSetDevice( 0 );
    cudaMalloc( &dev_buffer, BYTE_SIZE );
    kernel_on_dev1<<< SIZE, 1 >>>( dev_buffer );
    std::vector< real_t > host_buffer( SIZE );
    // sync and copy
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Kernel on device 1: " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    

    // on device 1
    cudaSetDevice( 1 );
    const int PEER_DEVICE_TO_ACCESS = 0;
    const int PEER_ACCESS_FLAGS = 0; // reserved for future use, must be zero
    cudaDeviceEnablePeerAccess( PEER_DEVICE_TO_ACCESS, PEER_ACCESS_FLAGS ); // <- enable current device(1) to access device 0
    kernel_on_dev2<<<  SIZE, 1 >>>( dev_buffer );
    // sync and copy
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Kernel on device 2: " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    cudaFree( dev_buffer );
    return 0;
}
