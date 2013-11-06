#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

//------------------------------------------------------------------------------
#define CCHECK(id) { \
    if(id != CUDA_SUCCESS) { \
        std::cerr << (__LINE__ - 1) << " CUDA ERROR " << id << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

typedef float real_t;

typedef std::vector< real_t > array_t;

// iota has been removed (why?) from STL long ago.
template< class FwdIt, class T > 
inline void iota( FwdIt begin, FwdIt end, T startVal ) {
    // compute increasing sequence into [begin, end)
    for (; begin != end; ++begin, ++startVal ) *begin = startVal;
}

//------------------------------------------------------------------------------
void mat_mul_test(const char* kernel_path,
                  const char* kernel_name,
                  int grid_dim_x,  int grid_dim_y,  int grid_dim_z,
                  int block_dim_x, int block_dim_y, int block_dim_z) {
    const int MATRIX_WIDTH = 1024; 
    const int MATRIX_HEIGHT = MATRIX_WIDTH; 
    const int VECTOR_SIZE = MATRIX_WIDTH;
    const int MATRIX_SIZE = MATRIX_WIDTH * MATRIX_HEIGHT;
    const int MATRIX_BYTE_SIZE = sizeof(real_t) * MATRIX_SIZE;
    const int VECTOR_BYTE_SIZE = sizeof(real_t) * VECTOR_SIZE;
   
    CCHECK(cuInit(0));
    array_t in_matrix_h(MATRIX_SIZE,  real_t(0));
    array_t in_vector_h(VECTOR_SIZE,  real_t(0));
    array_t out_vector_h(VECTOR_SIZE, real_t(0));
    CUdeviceptr in_matrix_d = 0;
    CUdeviceptr in_vector_d = 0;
    CUdeviceptr out_vector_d = 0;
    CCHECK(cuMemAlloc(&in_matrix_d, MATRIX_BYTE_SIZE));
    assert(in_matrix_d);
    CCHECK(cuMemAlloc(&in_vector_d, VECTOR_BYTE_SIZE));
    assert(in_vector_d);
    CCHECK(cuMemAlloc(&out_vector_d, VECTOR_BYTE_SIZE));
    assert(out_vector_d);
    iota(in_matrix_h.begin(), in_matrix_h.end(), real_t(0));
    iota(in_vector_h.begin(), in_vector_h.end(), real_t(0));
    CCHECK(cuMemcpy(in_matrix_d,  CUdeviceptr(&in_matrix_h[0]),  MATRIX_BYTE_SIZE));
    CCHECK(cuMemcpy(in_vector_d,  CUdeviceptr(&in_vector_h[0]),  VECTOR_BYTE_SIZE));
    CCHECK(cuMemcpy(out_vector_d, CUdeviceptr(&out_vector_h[0]), VECTOR_BYTE_SIZE));
    CUdevice device = CUdevice();
    CUcontext ctx = CUcontext();
    CCHECK(cuCtxCreate(&ctx, 0, device));
    CUmodule module = CUmodule();
    CCHECK(cuModuleLoad(&module, kernel_path));
    CUfunction fun = CUfunction();
    CCHECK(cuModuleGetFunction(&fun, module, kernel_name))
    std::vector< void* > kernel_params;
    kernel_params.push_back(&in_matrix_d);
    kernel_params.push_back((void*)(&MATRIX_WIDTH));
    kernel_params.push_back((void*)(&MATRIX_HEIGHT));
    kernel_params.push_back(&in_vector_d);
    kernel_params.push_back(&out_vector_d);
    cuLaunchKernel(fun, 
                   grid_dim_x, grid_dim_y, grid_dim_z,
                   block_dim_x, block_dim_y, block_dim_z,
                   0,//shared_mem_bytes
                   0, //stream
                   &kernel_params[0],
                   0);
    
    CCHECK(cuMemcpy(CUdeviceptr(&out_vector_h[0]), out_vector_d, VECTOR_BYTE_SIZE));
      
    // print first two and last elements
    std::cout << "vector[0]    = " << out_vector_h[ 0 ] << '\n';
    std::cout << "vector[1]    = " << out_vector_h[ 1 ] << '\n';
    std::cout << "vector[last] = " << out_vector_h.back() << std::endl;

    CCHECK(cuMemFree(in_matrix_d));
    CCHECK(cuMemFree(in_vector_d));
    CCHECK(cuMemFree(out_vector_d));
    CCHECK(cuModuleUnload(module));
    CCHECK(cuCtxDestroy(ctx));
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout << "usage: " << std::endl;
        std::cout << argv[0] << " <file path> <kernel name>" << std::endl;
    }
    const int grid_dim_x = 1;
    const int grid_dim_y = 1024;
    const int grid_dim_z = 1;
    const int block_dim_x = 1;
    const int block_dim_y = 256;
    const int block_dim_z = 1;
    mat_mul_test(argv[1], argv[2], grid_dim_x, grid_dim_y, grid_dim_z,
                 block_dim_x, block_dim_y, block_dim_z);
    return 0;
};