// #CUDA Training
//
// #Example 2.0 - sum vectors, launch gid size == array size
//
// #Author Ugo Varetto
//


//#include <cuda_runtime.h> // automatically added by nvcc
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

typedef float real_t;

// In this case the kernel assumes that the computation was started with enough
// threads to cover the entire domain. This is the preferred solution provided
// there are enough threads to cover the entire domain which might not be the
// case in case of a 1D grid layout (max number of threads = 512 threads per
// block x 65536  blocks = 2^25 = 32 Mi threads)
__global__ void sum_vectors(const real_t* v1, const real_t* v2, real_t* out,
                            size_t num_elements) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // since we assume that num threads >= num element we need to make sure we
    // do not write outside the range of the output buffer
    if (xIndex < num_elements) out[xIndex] = v1[xIndex] + v2[xIndex];
}

//------------------------------------------------------------------------------
int main(int, char**) {
    const int VECTOR_SIZE = 0x10000 + 1;            // vector size 65537
    const int SIZE = sizeof(real_t) * VECTOR_SIZE;  // total size in bytes
    const int THREADS_PER_BLOCK = 32;  // number of gpu threads per block

    // block size: the number of threads per block multiplied by the number of
    // blocks must be at least equal to NUMBER_OF_THREADS
    const int NUMBER_OF_BLOCKS =
        (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // if number of threads is not evenly divisable by the number of threads per
    // block we need an additional block; the above code can be rewritten as if(
    // NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) BLOCK_SIZE =
    // NUMBER_OF_THREADS / THREADS_PER_BLOCK; else BLOCK_SIZE =
    // NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1

    // host allocated storage; use std vectors to simplify memory management
    // and initialization
    std::vector<real_t> v1(VECTOR_SIZE, 1.f);    // initialize all elements to 1
    std::vector<real_t> v2(VECTOR_SIZE, 2.f);    // initialize all elements to 2
    std::vector<real_t> vout(VECTOR_SIZE, 0.f);  // initialize all elements to 0

    // gpu allocated storage
    real_t* dev_in1 = 0;  // vector 1
    real_t* dev_in2 = 0;  // vector 2
    real_t* dev_out = 0;  // result value
    cudaMalloc(&dev_in1, SIZE);

    cudaMalloc(&dev_in2, SIZE);
    cudaMalloc(&dev_out, SIZE);

    // copy data to GPU
    cudaMemcpy(dev_in1, &v1[0], SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_in2, &v2[0], SIZE, cudaMemcpyHostToDevice);

    // execute kernel with num threads >= num elements
    sum_vectors<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(dev_in1, dev_in2,
                                                         dev_out, VECTOR_SIZE);
    // read back result
    cudaMemcpy(&vout[0], dev_out, SIZE, cudaMemcpyDeviceToHost);

    // print first and last element of vector
    std::cout << "result: " << vout.front() << ".." << vout.back() << std::endl;

    // free memory
    cudaFree(dev_in1);
    cudaFree(dev_in2);
    cudaFree(dev_out);

    return 0;
}
