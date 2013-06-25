#pragma once

// CUDA Error handlers
// Author: Ugo Varetto


#include <cuda.h>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

//Handle error conditions with or without exceptions

//In CUDA every function returns a status code.
//Error conditions are signalled by a status code other than 'cudaSuccess'
//Textual information can be obtained from an error code through a call
//to cudaGetErrorString.
//Note that in CUDA kernel launches are asynchronous and it is therefore
//possible only to detect errors which cause a failed kernel launch, errors
//generated during kernel execution are not reported. 

//usage: 
// w/ exceptions:
//try {
//...
//  HANDLE_CUDA_ERROR( cudaMemcpy( ... ) );
//...
//  LAUNCH_CUDA_KERNEL( kernel<<<...>>>(...) )
//} catch( const std::exception& e ) {
//    std::cerr << "e.what()" << std::endl;
//}
//
// w/o exceptions:
// ...
// DIE_ON_CUDA_ERROR( cudaMemcpy( ... ) );
// ...
// DIE_ON_FAILED_KERNEL_LAUNCH( kernel<<<...>>>(...) )
 

//Note: 'inline' is required simply because the functions are defined inside
//       an include; not adding it violates the one definition rule when
//       multiple source files include this include file 

inline void HandleCUDAError( cudaError_t err,
                             const char *file,
                             int line,
                             const char* msg = 0 ) {
    if( err != cudaSuccess ) {
        std::ostringstream ss;
        ss << ( msg != 0 ? msg : "" ) << " File: " << file 
                                      << ", Line: " << line 
                                      << ", Error: " 
                                      << cudaGetErrorString( err );
        throw std::runtime_error( ss.str() );
    }
}

inline void DieOnCUDAError( cudaError_t err,
                            const char *file,
                            int line,
                            const char* msg = 0 ) {
    if( err != cudaSuccess ) {
      std::cerr << ( msg != 0 ? msg : "" ) 
                << " File: " 
                << file << ", Line: " << line << ", Error: " 
                << cudaGetErrorString( err ) << std::endl;
      exit( 1 );          
    }
}


#define CUDA_CHECK DIE_ON_CUDA_ERROR

#define HANDLE_CUDA_ERROR( err ) ( HandleCUDAError( err, __FILE__, __LINE__ ) )

#define DIE_ON_CUDA_ERROR( err ) ( DieOnCUDAError( err, __FILE__, __LINE__ ) )

// warning: since kernel execution is asynchronous this macro will only catch
//          errors resulting from a failed kernel launch but the error generated
//          during kernel execution
#define LAUNCH_CUDA_KERNEL( k ) \
    k; \
    HandleCUDAError(cudaGetLastError(), __FILE__, __LINE__, "(Kernel launch)"); \

#define DIE_ON_FAILED_KERNEL_LAUNCH( k ) \
    k; \
    DieOnCUDAError(cudaGetLastError(), __FILE__, __LINE__, "(Kernel launch)");
