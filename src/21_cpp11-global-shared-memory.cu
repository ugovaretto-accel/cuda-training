//Author: Ugo Varetto
//Example showing the use of C++11 features to access an array
//shared between host and device.
//Requires CUDA >= 6.5 and g++ 4.8
//compilation: nvcc -std=c++11 ...

#include <iostream>
#include <cstdlib>
#include <cassert>



//------------------------------------------------------------------------------
template < int i >
struct Params {
    template < typename HeadT, typename...TailT >
    static auto Get(const HeadT& h, const TailT&...t) 
           -> decltype(Params< i - 1 >::Get(t...)) {
        return Params< i - 1 >::Get(t...);
    }
};

template <>
struct Params< 0 > {
    template < typename HeadT, typename...TailT >
    static const HeadT& Get(const HeadT& h, const TailT&...t) { return h; }
};

template < int i, typename...Args >
auto Extract(const Args&...args) 
-> decltype(Params< i >::Get(args...)) {
    return Params< i >::Get(args...);
}

//------------------------------------------------------------------------------

template < typename H, typename...T >
__device__ auto Head(H h, T...t) -> H {
    return h;
}

template < typename T, typename... Args>
__global__ void Init(T* v, Args...args) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = Head(args...);
    v[idx] = i;
 }

int main(int, char**) {
    const int SIZE = 128;
    const int INIT_VALUE = 3;
    int* data = 0; //nullptr ?
    if(cudaMallocManaged(&data, SIZE * sizeof(int))
        != cudaSuccess) {
        std::cerr << "Error allocating memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    Init<<< 1, 128 >>>(data, INIT_VALUE);
    if(cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Sync error" << std::endl;
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i != SIZE; ++i) assert(data[i] == INIT_VALUE);
    std::cout << "PASSED" << std::endl;
    if(cudaFree(data) != cudaSuccess) {
        std::cerr << "Free error" << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}

