//Author: Ugo Varetto
//Example showing the use of C++11 features while also accessing an array
//shared between host and device throuh UVA/global shared memory.
//Requires CUDA >= 6.5 and g++ 4.8
//nvcc -std=c++11 -arch="sm_35" -Xcudafe "--diag_suppress=set_but_not_used"
//  to enable static_assert -DTRIGGER_ASSERT
//  to enable call to deleted method -DTRIGGER_DELETED_METHOD
//  to enable final method override -DTRIGGER_FINAL
//warning about unused instance is suppressed, instance is declared then
//used to initialize separate object.
//Tested C++11 features
//- auto
//- decltype
//- variadic templates
//- lambda with capture by value
//- nullptr
//- r-value references
//- scoped and based enums
//- static_assert
//- deleted methods
//- default constructors: works but __device__ prefix is required
//- long long int
//- constexpr: does work as template argument as e.g. argument to
//             static_assert, does not work as size of POD array
//- range based for loop
//- user defined literals
//- initializer list
//- sizeof on class member
//- alignof
//- alignas: NOT SUPPORTED
//- unrestricted unions
//- right angle brackets
//- template aliases
//- new string literals
//- attributes: [[attribute]] is parsed correctly and does not result in 
//              compilation errors
//- (STL) type traits
//- (stdlib) <cstdint> types

#include <iostream>//cout
#include <cstdlib> //exit
#include <cassert> //assert
#include <cstdint> //int8_t
#include <type_traits> //is_union

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


struct FinalBase {
    __device__ virtual void FinalMethod() final {}
};

//Allow calling method with float type only, 
//no automatic conversions allowed
struct CallableWithFloatOnly : FinalBase {
    __device__ CallableWithFloatOnly(float f) : v(f) {}
    __device__ CallableWithFloatOnly() = default;
    __device__ CallableWithFloatOnly(const CallableWithFloatOnly&) = default;
    __device__ void Call(float f) {
        printf("CallableWithFloatOnly::Call(%f)\n", f);
    }
    template < typename T >
    void Call(T) = delete; //no need to prefix with __device__ 
                           //for deleted method
#ifdef TRIGGER_FINAL
    __device__ void FinalMethod() override {} //FAIL
#endif
    float v = 0.0f; //required to trigger compilation for constructors
};

//constexpr:__device__ required
 __device__ constexpr size_t ArraySize() { return size_t(5); }

//range based for loops
//In order to have range-based for loops working with non-STL collections
//it is required to have proper implementations of the begin and end functions.
//Name dependent lookup is used for selecting the right begin/end function but
//in the case of plain arrays the selected namespace is std and it is therefore
//required to have begin/end implemented in the std namespace (+__device__ for
//CUDA).
//"6.5.4 [stmt.ranged] p1
//...otherwise, begin-expr and end-expr are begin(__range) and end(__range),
//respectively, where begin and end are looked up with argument-dependent lookup
//(3.4.2). For the purposes of this name lookup, namespace std is an associated
//namespace."
namespace std{
__device__ int* begin(int* ptr) { return ptr; }
__device__ int* end(int* ptr) { return ptr + ArraySize(); }
}

//call with constexpr template parameter
template < size_t s >
__device__ void Foo() {
    printf("constexpr: Foo<%llu>()\n", s);
}

//user defined literals
__device__
size_t operator "" _l(const char* , size_t s) { return s; }

//unrestricted unions
//use with union
struct Int {
    int i_ = 0;
    __host__ __device__ operator int() const { return i_; }
    __host__ __device__ Int(int i) : i_(i) {}
    __host__ __device__ Int() = default;
    __host__ __device__ Int(const Int&) = default;
};
template < typename U >
union UnrestrictedUnion {
    int anInt;
    U m;
    __host__ __device__ UnrestrictedUnion() :  m(U()) {}
};

//template aliases (host only)
template < typename A, typename B >
class TClass {};
template < typename T >
using STClass = TClass< T, int >;
using SSTClass = STClass< float >;

//Kernel implementation
template < typename T, typename... Args>
__global__ void Init(T* v, Args...args) {
    //nullptr
    assert(v != nullptr);
    //static_assert
    static_assert(sizeof...(Args) > 0, "Empty argument list");
    static_assert(ArraySize() > 0, "Zero array size");
    static_assert(sizeof(long long int) >= 8, "Non compliant 'long long' size");
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //scoped and based enums
    enum class Enum : int8_t { A = 65, B, C };
    if(idx == 0) {
        printf("Scoped enum 'A' = %c, 'B' = %c, 'C' = %c\n",
               static_cast< int >(Enum::A),
               static_cast< int >(Enum::B),
               static_cast< int >(Enum::C));
    }
    //lambda with capture
    if(idx > 0 && idx < 20) 
        Invoke([idx](){ printf("Hi from (GPU) thread %d\n", idx); });
    //variadic templates
    //WARNING: in standard C++11 code compiled with
    //clang 3.4 and gcc 4.8.2 the ", Args..." part is not required
    auto i = Extract< 0, Args... >(args...);
    //deleted methods and initializer list
    if(idx == 0) {
        CallableWithFloatOnly cf = {2.0f};
        CallableWithFloatOnly cf2(cf);
        assert(cf2.v == 2.0f);
        cf2.Call(Extract< 1, Args... >(args...));
    }
    //final
    if(idx == 0) {
        CallableWithFloatOnly cf;
        cf.FinalMethod(); 
     }
    //r-value references
    if(idx == 0) {
        Ref(Head(args...));
    }
    //constexpr
    if(idx == 1) Foo< ArraySize() >();
    //WARNING: constexpr expression ArraySize() cannot be used to specify
    //the size of the array; the following code results in a compile time
    //error: "error: variable "array" may not be initialized"
    //int array[ArraySize()] = {1, 2, 3, 4, 5};
    int array[] = {1, 2, 3, 4, 5};
    //range based for loop
    int sum = 0;
    for(auto& j: array) {
        sum += j;
    }
    assert(sum == 15);
    //user defined literal
    if(idx == 0) {
        const size_t length = "12345"_l;
        printf("length of '12345' is %llu\n", length);
    }
    //sizeof on type member
    assert(sizeof(CallableWithFloatOnly::v) == sizeof(float));
    //alignof and alignas
    const size_t floatAlignment = alignof(float);
#ifdef TEST_ALIGNAS
    alignas(float) const char carray[3];
    assert(size_t(&carray[0]) % alignof(float) == 0);
#endif
    if(idx == 0) {
        printf("float alignment = %llu\n", floatAlignment);
    }
    //unrestricted unions
    UnrestrictedUnion< Int > uu;
    uu.m = Int(i);
    //initialize array
    v[idx] = uu.anInt;
    //right angle bracket and type_traits
    static_assert(std::is_union< UnrestrictedUnion< int >>::value, "Not a union!");
    //new string literals
const char* text =R"text(
-------------------
Some formatted text
with a newline
------------------
)text";
    if(idx == 0) printf("%s", text);
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
#ifndef TRIGGER_ASSERT
#ifndef TRIGGER_DELETED_METHOD
    Init<<< 1, 128 >>>(data, INIT_VALUE, 2.0f); //call with float
#else
    Init<<< 1, 128 >>>(data, INIT_VALUE, 2.0); //call with double -> failure
#endif
#else
    Init<<< 1, 128 >>>(data);
#endif
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


