// #CSCS CUDA Training 
//
// #Example 16 - copy memory with cudaMemcpy3D
//
// #Author Ugo Varetto
//
// #Goal: copy data from a subregion of a 3D grid on the GPU
//        to another grid on the GPU and from the output grid to a grid on
//        the CPU using the same CUDA function for all copy operations
//
//
// #Rationale: extracting/inserting data from/to regular 3D grids is a common
//             task performed in distributed stencil computations; CUDA provides
//             facilities to ease such task through the cudaMemcpy3D function;
//             the CUDA function for multidimensional copy work between any pair
//             of devices:
//             - GPU device to/from same GPU device
//             - GPU device to/from different GPU device
//             - GPU device to/from host memory
//              
//
// #Solution: invoke cudaMemcpy3D specifying the various 3D parameters such as
//            offsets and extents as
//            (*row byte size*, number of rows, number of slices).
//            use two configurations: one for deice to device and the other for
//            device to host
//
// #Code: 
//        1) allocate input and output grid on cpu
//        2) init cpu grid with data
//        3) allocate input and output grid on GPU and init input
//           grid by copying from cpu input grid
//        3) initialize cudaMemcpy3DParms structure with details
//           of data exchange(io pointers, offsets and extent)
//        4) invoke cudaMemcpy3D for device to device copy
//        5) invoke cudaMemcpy3D for device to host copy
//        5) check that results are the same as the ones found in the
//           input sub-region that was extracted
//
// #Compilation: nvcc -arch=sm_13 17_memcpy3d.u -o 17_memcpy_3d 
//
// #Execution: ./17_memcpy_3d
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better


#include <iostream>
#include <vector>
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

bool check_grid(real_t* grid, 
                int xoff,
                int yoff,
                int zoff,
                int width,
                int height,
                int depth) {
    const int offset = xoff + yoff + zoff;
    for(int k = 0; k != depth; ++k) {
        for(int j = 0; j != height; ++j) {
            for(int i = 0; i != width; ++i) {
                if(grid[coord_to_idx(i, j, k, width, height)] 
                   != i + j + k + offset ) return false;
            }
        }
    }
    return true;
}

#define CHECK DIE_ON_CUDA_ERROR

int main(int, char**) {

    const int in_width = 100; //faster if width * sizeof(real_t) = k * 512 with k a positive integer
    const int in_height = 200;
    const int in_depth = 300;
    const int in_size = in_width * in_height * in_depth;
    const int in_byte_size = in_size * sizeof(real_t);
    const int in_byte_row = in_width * sizeof(real_t);
    const int xoffset = 1;
    const int xoffset_bytes = xoffset * sizeof(real_t);
    const int yoffset = 1;
    const int zoffset = 1;
    const int out_width = in_width - xoffset; //faster if width * sizeof(real_t) = k * 512 with k a positive integer
    const int out_height = in_height - yoffset;
    const int out_depth = in_depth - zoffset;
    const int out_size = out_width * out_height * out_depth;
    const int out_byte_size = out_size * sizeof(real_t);
    const int out_byte_row = out_width * sizeof(real_t);
    
    std::vector< real_t > h_grid_in(in_size, 0);
    std::vector< real_t > h_grid_out(out_size, 0);
    init_grid(&h_grid_in[0], in_width, in_height, in_depth);
    
    real_t* d_grid_in = 0;
    real_t* d_grid_out = 0;

    //Extents/positions for linear memory are ALWAYS specified as:
    // row width IN BYTES, column height IN ELEMENTS, depth IN ELEMENTS
    // fastest performance for 512 byte-aligned allocations 

    CHECK(cudaMalloc(&d_grid_in, in_byte_size));
    CHECK(cudaMemcpy(d_grid_in, &h_grid_in[0], in_byte_size, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_grid_out, out_byte_size));
    
    cudaMemcpy3DParms memcpyParams;

    cudaPitchedPtr inptr = make_cudaPitchedPtr(d_grid_in, in_byte_row, in_width, in_height);
    cudaPitchedPtr outptr = make_cudaPitchedPtr(d_grid_out, out_byte_row, out_width, out_height);

    //configure for device to device copy
    memcpyParams.srcArray = 0;
    memcpyParams.srcPos = make_cudaPos(xoffset_bytes, yoffset, zoffset);
    memcpyParams.srcPtr = inptr;
    memcpyParams.dstArray = 0;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = outptr;
    memcpyParams.extent = make_cudaExtent(out_byte_row, out_height, out_depth);
    memcpyParams.kind = cudaMemcpyDeviceToDevice;

    CHECK(cudaMemcpy3D(&memcpyParams));

    cudaPitchedPtr host_outptr =
        make_cudaPitchedPtr(&h_grid_out[0],
                            out_byte_row, out_width, out_height);
    host_outptr.pitch = out_byte_row;    
    
    //configure for device to host copy
    memcpyParams.srcArray = 0;
    memcpyParams.srcPos = make_cudaPos(0,0,0);
    memcpyParams.srcPtr = outptr;
    memcpyParams.dstArray = 0;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = host_outptr;
    memcpyParams.extent = make_cudaExtent(out_byte_row, out_height, out_depth);
    memcpyParams.kind = cudaMemcpyDeviceToHost;
 
    CHECK(cudaMemcpy3D(&memcpyParams));

    std::cout << std::boolalpha 
              << "Copied: " 
              << check_grid(&h_grid_out[0], xoffset, yoffset, zoffset,
                            out_width, out_height, out_depth)
              << std::endl;

    CHECK(cudaFree(d_grid_in));
    CHECK(cudaFree(d_grid_out));
    CHECK(cudaDeviceReset());
    return 0;
} 