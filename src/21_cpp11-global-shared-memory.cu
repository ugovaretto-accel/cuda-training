//Author: Ugo Varetto
//Example showing the use of C++11 features while also accessing an array
//shared between host and device throuh UVA/global shared memory.
//Requires CUDA >= 6.5 and g++ 4.8
//compilation: nvcc -std=c++11 ...
//Tested C++11 features
//- auto
//- decltype
//- variadic templates
//- lambda with capture by value
//- nullptr
//- r-value references
//- scoped and based enums

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cstdint>

//------------------------------------------------------------------------------
template < int i >
struct Params {
    template < typename HeadT, typename...TailT >
    __device__ static auto Get(const HeadT& h, const TailT&...t) 
           -> decltype(Params< i - 1 >::Get(t...)) {
        return Params< i - 1 >::Get(t...);
    }
};

template <>
struct Params< 0 > {
    template < typename HeadT, typename...TailT >
    __device__ static const HeadT& Get(const HeadT& h, const TailT&...t) { 
        return h;
    }
};

//Extract element at position i from variadic argument list
template < int i, typename...Args >
__device__ auto Extract(const Args&...args) 
-> decltype(Params< i >::Get(args...)) {
    return Params< i >::Get(args...);
}

//------------------------------------------------------------------------------
//Return first element in vararg list
template < typename H, typename...T >
__device__ auto Head(H h, T...t) -> H {
    return h;
}
//Invoke function (object) passes as argument
template < typename F >
__device__ void Invoke(F f) { f(); }
//R-value reference
template < typename T > 
__device__ void Ref(T&&) {
    printf("R-value reference\n");
}
template < typename T > 
__device__ void Ref(const T&) {
    printf("Const reference\n");
}
//Kernel implementation
template < typename T, typename... Args>
__global__ void Init(T* v, Args...args) {
    //nullptr
    assert(v != nullptr);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //scoped and based enums
    enum class Enum : int8_t { A = 65, B, C };
    if(idx == 0) {
        printf("Scoped enum 'A' = %c, 'B' = %c, 'C' = %c\n",
               static_cast< int >(Enum::A),
               static_cast< int >(Enum::B),
               static_cast< int >(Enum::C));
    }
    //lmbda with capture
    if(idx > 0 && idx < 20) 
        Invoke([idx](){ printf("Hi from (GPU) thread %d\n", idx); });
    //variadic templates
    //WARNING: in standard C++11 code compiled with
    //clang 3.4 and gcc 4.8.2 the ", Args..." part is not required
    auto i = Extract< 0, Args... >(args...);
    //r-value references
    if(idx == 0) {
        Ref(Head(args...));
    }
    //initialize array
    v[idx] = i;
 }

//------------------------------------------------------------------------------
int main(int, char**) {
    const int SIZE = 128;
    const int INIT_VALUE = 3;
    int* data = nullptr;
    //Allocate managed memory region: both GPU and CPU can access the memory
    //using the same base pointer
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

