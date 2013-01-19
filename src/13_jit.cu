// #CSCS CUDA Training 
//
// #Example 13 - jit compilation, kernel to be compiled to ptx
//               and loaded from example 13 
//
// #Author Ugo Varetto
//
// #Compilation: nvcc -ptx -arch=sm_13 13_jit.cu -> generates 13_jit.ptx

typedef float real_t;

extern "C"
 __global__ void sum_vectors( const real_t* v1, const real_t* v2, real_t* out, int num_elements ) {
     // compute current thread id
     const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
     // since we assume that num threads >= num element we need to make sure we do not write outside the
     // range of the output buffer 
     if( xIndex < num_elements ) out[ xIndex ] = v1[ xIndex ] + v2[ xIndex ];
 }
