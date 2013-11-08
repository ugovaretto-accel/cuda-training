//Author: Ugo Varetto
//New (5.x) driver API example
//Compilation on Cray XK7 and XC-30 with gnu:
//g++ ../../src/20_driver_api_linker.cpp \
//-I /opt/nvidia/cudatoolkit/default/include \
//-L /opt/cray/nvidia/default/lib64 -lcuda
//Load, compile and link any number of ptx files at run-time then execute
//a kernel.

#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <string>

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
void build(CUmodule& module,
           CUfunction& kernel,
           const std::vector< std::string >& files,
           const char* kernel_name) {


    CUjit_option options[] = {CU_JIT_WALL_TIME,
                              CU_JIT_INFO_LOG_BUFFER,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_LOG_VERBOSE};
    float walltime = 0.f;
    const unsigned bufsize = 0x10000;
    char error_buf[bufsize] = "";
    char log_buf[bufsize] = "";
    const int verbose = 1;                          
    void* option_values[] = {(void*) &walltime,
                             (void*) log_buf, 
                             (void*) bufsize,
                             (void*) error_buf,
                             (void*) bufsize,
                             (void*) verbose};

    void* compiled_code = 0;
    size_t compiled_size = 0;
    int status = CUDA_SUCCESS - 1;
      
    CUlinkState link_state = CUlinkState();
    
    const int num_options = sizeof(options) / sizeof(CUjit_option);

    // Create a pending linker invocation
    CCHECK(cuLinkCreate(num_options,
                        options, option_values, &link_state));

    for(std::vector< std::string >::const_iterator i = files.begin();
        i != files.end();
        ++i) {
        status = cuLinkAddFile(link_state,
                             CU_JIT_INPUT_PTX, 
                             i->c_str(),
                             0, //num options
                             0, //options,
                             0); //option values
    }


    if( status != CUDA_SUCCESS ) {
        std::cerr << "PTX Linker Error:\n"<< error_buf << std::endl;
        exit(EXIT_FAILURE);
    }

    // Complete the linker step: compiled_code is filled with executable code
    //???: what do I do with the returned data ? can/should I delete it ?
    CCHECK(cuLinkComplete(link_state, &compiled_code, &compiled_size));
    assert(compiled_size > 0);
    assert(compiled_code);

    std::cout << "CUDA Link Completed in " << walltime << " ms\n"
              << log_buf << std::endl; 

    CCHECK(cuModuleLoadData(&module, compiled_code));

    CCHECK(cuModuleGetFunction(&kernel, module, kernel_name));

    CCHECK(cuLinkDestroy(link_state));
}


//------------------------------------------------------------------------------
void mat_mul_test(const std::vector< std::string >& file_paths,
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
    CUfunction fun = CUfunction(); 
   
    build(module, fun, file_paths, kernel_name);
       
    void* kernel_params[] = {&in_matrix_d,
                             (void *)(&MATRIX_WIDTH),
                             (void *)(&MATRIX_HEIGHT),
                             &in_vector_d,
                             &out_vector_d};
    CCHECK(cuLaunchKernel(fun, 
                    grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    0,//shared_mem_bytes
                    0,//stream
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
        std::cout << argv[0] << " <kernel name> <file paths>..."
                  << std::endl;
        exit(EXIT_FAILURE);          
    }
    const int grid_dim_x = 4;
    const int grid_dim_y = 1;
    const int grid_dim_z = 1;
    const int block_dim_x = 256;
    const int block_dim_y = 1;
    const int block_dim_z = 1;
    const int size = 1024;

    std::vector< std::string > paths;
    for(int i = 2; i != argc; ++i) {
        paths.push_back(argv[i]);
    }

    mat_mul_test(paths, argv[1], size,
                 grid_dim_x, grid_dim_y, grid_dim_z,
                 block_dim_x, block_dim_y, block_dim_z);
    return 0;
};