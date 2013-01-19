// #CSCS CUDA Training 
//
// #Example 13 - jit compilation
//
// #Author Ugo Varetto
//
// #Goal: load and execute a kernel at run-time
//
// #Rationale: shows how to load and execute a function and ptx/cubin file 
//
// #Solution: use the cuda driver api to load a module and execute the kernel;
//            the function in the pre-compiled file must be declared as extern "C"
//            in order to be called with its name; alternatively it is possible to
//            use the mangled name by looking at the function name in the file  
//          
// #Code: 
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host to device
//        4) load module (ptx or cubin) file
//        5) get function to call from function name
//        6) pack pointers to kernels parameters into array
//        7) invoke kernel with grid layout info and parameters 
//        6) consume data (in this case print result)
//        7) free memory
//        
// #Compilation: use platform C++ compiler to compile file and link cudart and cuda libs;
//               pre-compile to ptx kernel; see 13_jit.cu. 
//
// #Execution: ./13_jit 13_jit.ptx sum_vectors
//
// #Note: kernel invocations are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied       



#include <cuda.h> // <- driver API cuXXX
#include <cuda_runtime.h> // <- runtime API cudaXXX
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

typedef float real_t;


//------------------------------------------------------------------------------
int main( int argc, char** argv  ) {
    
    if( argc != 3 ) {
        std::cout << "usage: " << argv[ 0 ] << " <ptx file name> <kernel function name>" << std::endl;
        return 1;
    }
    
    const int VECTOR_SIZE = 0x10000;// + 1; //vector size 65537
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
    cudaMalloc( &dev_in1, SIZE ); // <- first call into CUDA automatically creates context!
    cudaMalloc( &dev_in2, SIZE );
    cudaMalloc( &dev_out, SIZE  );
    
    // copy data to GPU
    cudaMemcpy( dev_in1, &v1[ 0 ], SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_in2, &v2[ 0 ], SIZE, cudaMemcpyHostToDevice );

//=========== BEGIN CUDA DRIVER API
    // use CUDA driver API function to load and execute the kernel;
    // when using the CUDA driver API a valid context must be available,
    // in this case a context is available because any CUDA runtime API
    // function such as cudaMemcpy creates and initializes a context 
    // automatically if no active context is found
    CUmodule module;
    CUresult status = cuModuleLoad( &module, argv[ 1 ] );
    if( status != CUDA_SUCCESS ) {
        std::cout << "Cannot load module " << argv[ 1 ] << std::endl;
        return 1;
    }
    CUfunction function;
    status = cuModuleGetFunction( &function, module, argv[ 2 ] );
    if( status != CUDA_SUCCESS ) {
        std::cout << "Cannot retrieve function " << argv[ 2 ] << " attributes" << std::endl;
        return 1;
    } 
   
    // pack pointers to kernel parameters into array
    std::vector< void* > kernelParams;
    kernelParams.push_back( &dev_in1 );
    kernelParams.push_back( &dev_in2 );
    kernelParams.push_back( &dev_out );
    // why cannot they use const for immutable data ? so that we can use a const array of const pointers ?
    kernelParams.push_back( const_cast< int* >( &VECTOR_SIZE ) );

    const size_t sharedMemSize = 0;
    const CUstream stream = 0;
   
        // equivalent to
    // sum_vectors<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>( dev_in1, dev_in2, dev_out, VECTOR_SIZE );
    status = cuLaunchKernel( function, NUMBER_OF_BLOCKS, 1, 1,
                             THREADS_PER_BLOCK, 1, 1,
                             sharedMemSize, stream,
                             &kernelParams[ 0 ], 0 );
    if( status != CUDA_SUCCESS ) {
        std::cout << "Kernel launch failure " << status << std::endl;
        return 1;        
    }
//=========== END CUDA DRIVER API                        
    

    // read back result 
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    
    // print first and last element of vector
    std::cout << "result: " << vout.front() << ".." << vout.back() << std::endl;

    // free memory
    cudaFree( dev_in1 );
    cudaFree( dev_in2 );
    cudaFree( dev_out );

    return 0;
}
