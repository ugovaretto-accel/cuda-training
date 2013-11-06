//Author: Ugo Varetto
//New (5.x) driver API example
//Compilation on Cray XK7 and XC-30 with gnu:
//g++ ../../src/19_driver_api.cpp \
//-I /opt/nvidia/cudatoolkit/default/include \
//-L /opt/cray/nvidia/default/lib64 -lcuda

#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

//------------------------------------------------------------------------------
#define CCHECK(id) { \
    if(id != CUDA_SUCCESS) { \
        std::cerr << __LINE__ << " CUDA ERROR " << id << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

typedef float real_t;

typedef std::vector< real_t > array_t;

//------------------------------------------------------------------------------
void mat_mul_test(const char* kernel_path,
                  const char* kernel_name,
                  int size, 
                  int grid_dim_x,  int grid_dim_y,  int grid_dim_z,
                  int block_dim_x, int block_dim_y, int block_dim_z) {
    const int MATRIX_WIDTH = size; 
    const int MATRIX_HEIGHT = MATRIX_WIDTH; 
    const int VECTOR_SIZE = MATRIX_WIDTH;
    const int MATRIX_SIZE = MATRIX_WIDTH * MATRIX_HEIGHT;
    const int MATRIX_BYTE_SIZE = sizeof(real_t) * MATRIX_SIZE;
    const int VECTOR_BYTE_SIZE = sizeof(real_t) * VECTOR_SIZE;
   
    CCHECK(cuInit(0));
    array_t in_matrix_h(MATRIX_SIZE,  real_t(1));
    array_t in_vector_h(VECTOR_SIZE,  real_t(1));
    array_t out_vector_h(VECTOR_SIZE, real_t(0));
    CUdeviceptr in_matrix_d = 0;
    CUdeviceptr in_vector_d = 0;
    CUdeviceptr out_vector_d = 0;
    CUdevice device = CUdevice();
    CUcontext ctx = CUcontext();
    CCHECK(cuCtxCreate(&ctx, 0, device));
    CCHECK(cuMemAlloc(&in_matrix_d, MATRIX_BYTE_SIZE));
    assert(in_matrix_d);
    CCHECK(cuMemAlloc(&in_vector_d, VECTOR_BYTE_SIZE));
    assert(in_vector_d);
    CCHECK(cuMemAlloc(&out_vector_d, VECTOR_BYTE_SIZE));
    assert(out_vector_d);
    CCHECK(cuMemcpy(in_matrix_d,  CUdeviceptr(&in_matrix_h[0]),
           MATRIX_BYTE_SIZE));
    CCHECK(cuMemcpy(in_vector_d,  CUdeviceptr(&in_vector_h[0]),
           VECTOR_BYTE_SIZE));
    CCHECK(cuMemcpy(out_vector_d, CUdeviceptr(&out_vector_h[0]),
           VECTOR_BYTE_SIZE));
    CUmodule module = CUmodule();
    CCHECK(cuModuleLoad(&module, kernel_path));
    CUfunction fun = CUfunction();
    CCHECK(cuModuleGetFunction(&fun, module, kernel_name))
    void* kernel_params[] = {&in_matrix_d,
                             (void *)(&MATRIX_WIDTH),
                             (void *)(&MATRIX_HEIGHT),
                             &in_vector_d,
                             &out_vector_d};
    CCHECK(cuLaunchKernel(fun, 
                    grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    0,//shared_mem_bytes
                    0, //stream
                    kernel_params,
                    0));
    
    CCHECK(cuMemcpy(CUdeviceptr(&out_vector_h[0]), out_vector_d,
                    VECTOR_BYTE_SIZE));
      
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
    const int grid_dim_x = 4;
    const int grid_dim_y = 1;
    const int grid_dim_z = 1;
    const int block_dim_x = 256;
    const int block_dim_y = 1;
    const int block_dim_z = 1;
    const int size = 1024;
    mat_mul_test(argv[1], argv[2], size, grid_dim_x, grid_dim_y, grid_dim_z,
                 block_dim_x, block_dim_y, block_dim_z);
    return 0;
};