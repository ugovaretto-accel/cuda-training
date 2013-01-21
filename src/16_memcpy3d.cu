// #CSCS CUDA Training 
//
// #Example 16 - copy memory with cudaMemcpy3D
//
// #Author Ugo Varetto
//
// #Goal: copy data between two 3D grids on the GPU
//
// #Rationale: extracting/inserting data from/to regular 3D grids is a common
//             task performed in distributed stencil computations; CUDA provides
//             facilities to ease such task through the cudaMemcpy3D function
//              
//
// #Solution: invoke cudaMemcpy3D specifying the various 3D parameters such as
//            offsets and extents as
//            (*row byte size*, number of rows, number of slices)
//
// #Code: 
//        1) allocate input and output grid on cpu
//        2) init cpu grid with data
//        3) allocate input and output grid on GPU and init input
//           grid by copying from cpu input grid
//        3) initialize cudaMemcpy3DParms structure with details
//           of data exchange(io pointers, offsets and extent)
//        4) invoke cudaMemcpy3D
//        5) copy data back to cpu and check that results are the same
//           as input grid
//
// #Compilation: nvcc -arch=sm_13 16_memcpy3d.u -o 16_memcpy_3d 
//
// #Execution: ./16_memcpy_3d
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better


#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "cuda_error_handler.h"

typedef double real_t;

int coord_to_idx(int x, int y, int z, int row_stride, int col_stride) {
    return x + row_stride * (y + col_stride * z);
}

void init_grid(real_t* grid, 
               int width,
               int height,
               int depth) {
    for(int k = 0; k != depth; ++k) {
        for(int j = 0; j != height; ++j) {
            for(int i = 0; i != width; ++i) {
                grid[coord_to_idx(i, j, k, width, height)] = i + j + k;
            }
        }
    }
}

#define CHECK DIE_ON_CUDA_ERROR

int main(int, char**) {

    const int width = 100; //faster if width * sizeof(real_t) = k * 512 with k a positive integer
    const int height = 200;
    const int depth = 300;
    const int size = width * height * depth;
    const int byte_size = size * sizeof(real_t);
    const int byte_row = width * sizeof(real_t);

    std::vector< real_t > h_grid_in(size, 0);
    std::vector< real_t > h_grid_out(size, 0);
    init_grid(&h_grid_in[0], width, height, depth);
    
    real_t* d_grid_in = 0;
    real_t* d_grid_out = 0;

    //Extents/positions for linear memory are ALWAYS specified as:
    // row width IN BYTES, column height IN ELEMENTS, depth IN ELEMENTS
    // fastest performance for 512 byte-aligned allocations 

    CHECK(cudaMalloc(&d_grid_in, byte_size));
    CHECK(cudaMemcpy(d_grid_in, &h_grid_in[0], byte_size, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_grid_out, byte_size));
    
    cudaPitchedPtr inptr = make_cudaPitchedPtr(d_grid_in, byte_row, width, height);
    cudaPitchedPtr outptr = make_cudaPitchedPtr(d_grid_out, byte_row, width, height);

    cudaMemcpy3DParms memcpyParams;
    memcpyParams.srcArray = 0;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = inptr;
    memcpyParams.dstArray = 0;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = outptr;
    memcpyParams.extent = make_cudaExtent(byte_row, height, depth);
    memcpyParams.kind = cudaMemcpyDeviceToDevice;

    CHECK(cudaMemcpy3D(&memcpyParams));

    CHECK(cudaMemcpy(&h_grid_out[0], d_grid_out, byte_size, cudaMemcpyDeviceToHost));

    std::cout << std::boolalpha 
              << "Copied: " 
              << std::equal(h_grid_in.begin(), h_grid_in.end(), h_grid_out.begin())
              << std::endl;

    CHECK(cudaFree(d_grid_in));
    CHECK(cudaFree(d_grid_out));
    CHECK(cudaDeviceReset());
    return 0;
} 