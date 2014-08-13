#include <iostream>
#include <cstdlib>
#include <cassert>

template < typename T, typename... Args>
__global__ void Init(T* v, Args...) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    v[idx] = idx;
 }
int main(int, char**) {
    const int SIZE = 128;
    int* data = 0; //nullptr ?
    if(cudaMallocManaged(&data, SIZE * sizeof(int))
        != cudaSuccess) {
        std::cerr << "Error allocating memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    Init<<< 1, 128 >>>(data);
    if(cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Sync error" << std::endl;
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i != SIZE; ++i) assert(data[i] == i);
    std::cout << "PASSED" << std::endl;
    if(cudaFree(data) != cudaSuccess) {
        std::cerr << "Free error" << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}

