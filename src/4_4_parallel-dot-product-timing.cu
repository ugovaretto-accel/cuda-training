//Author: Ugo Varetto
//Parallel dot product with timing. Link with librt (-lrt)

//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <numeric>
#include <ctime>

typedef double real_t;

const size_t BLOCK_SIZE = 1024;

//------------------------------------------------------------------------------
double time_diff_ms(const timespec& start, const timespec& end) {
    return end.tv_sec * 1E3 +  end.tv_nsec / 1E6
           - (start.tv_sec * 1E3 + start.tv_nsec / 1E6);  
}



__global__ void partial_dot( const real_t* v1, const real_t* v2, real_t* out, int N ) {
    __shared__ real_t cache[ BLOCK_SIZE ];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[ threadIdx.x ] = 0.f;
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }    
    __syncthreads(); // required because later on the current thread is accessing
                     // data written by another thread
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; //not sure bitwise operations are actually faster
    }

    if( threadIdx.x == 0 ) out[ blockIdx.x ] = cache[ 0 ];
}

real_t dot( const real_t* v1, const real_t* v2, int N ) {
    real_t s = 0;
    for( int i = 0; i != N; ++i ) {
        s += v1[ i ] * v2[ i ];
    }
    return s;
}

real_t dot_block( const real_t* v1, const real_t* v2, int N, int block_size ) {
    std::vector< real_t > b1(block_size);
    std::vector< real_t > b2(block_size);
    real_t s = 0;
    for( int i = 0; i < N; i += block_size ) {
        std::copy(v1 + i, v1 + i + block_size, b1.begin());
        std::copy(v2 + i, v2 + i + block_size, b2.begin());
        s += dot(&b1[0], &b2[0], block_size);
    }
    return s;
}

__global__ void init_vector( real_t* v, int N ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while( i < N ) {
        v[ i ] = 1.0f;//real_t( i ) / 1000000.f;
        i += gridDim.x * blockDim.x;
    } 
}


//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
    
    const size_t ARRAY_SIZE = 1024 * 1024 * 256; //1Mi elements
    const int THREADS_PER_BLOCK = 1024; 
    const int BLOCKS = ARRAY_SIZE / THREADS_PER_BLOCK;//512;
    const size_t SIZE = ARRAY_SIZE * sizeof( real_t );
    
    // device storage
    real_t* dev_v1 = 0; // vector 1
    real_t* dev_v2 = 0; // vector 2
    real_t* dev_vout = 0; // partial redution = number of blocks
    cudaMalloc( &dev_v1,  SIZE );
    cudaMalloc( &dev_v2,  SIZE );
    cudaMalloc( &dev_vout, BLOCKS * sizeof( real_t ) );

    // host storage
    std::vector< real_t > host_v1( ARRAY_SIZE );
    std::vector< real_t > host_v2( ARRAY_SIZE );
    std::vector< real_t > host_vout( BLOCKS );

    // initialize vector 1 with kernel; much faster than using for loops on the cpu
    init_vector<<< BLOCKS, THREADS_PER_BLOCK  >>>( dev_v1, ARRAY_SIZE );
    cudaMemcpy( &host_v1[ 0 ], dev_v1, SIZE, cudaMemcpyDeviceToHost );
    // initialize vector 2 with kernel; much faster than using for loops on the cpu
    init_vector<<< BLOCKS, THREADS_PER_BLOCK  >>>( dev_v2, ARRAY_SIZE );
    cudaMemcpy( &host_v2[ 0 ], dev_v2, SIZE, cudaMemcpyDeviceToHost );
    
    timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);   
    // execute kernel
    partial_dot<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_v1, dev_v2, dev_vout, ARRAY_SIZE );
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &e);
    const double elapsed = time_diff_ms(s, e);
         
    // copy output data from device(gpu) to host(cpu)
    clock_gettime(CLOCK_MONOTONIC, &s);
    cudaMemcpy( &host_vout[ 0 ], dev_vout, BLOCKS * sizeof( real_t ), cudaMemcpyDeviceToHost );
    clock_gettime(CLOCK_MONOTONIC, &e);
    const double transferTime = time_diff_ms(s, e);

    clock_gettime(CLOCK_MONOTONIC, &s);
    const real_t device_dot = std::accumulate( host_vout.begin(), host_vout.end(), real_t( 0 ) );  
    clock_gettime(CLOCK_MONOTONIC, &e);
    const double acc = time_diff_ms(s, e);

    //dot product on host
    clock_gettime(CLOCK_MONOTONIC, &s);
    //const real_t host_dot = std::inner_product(host_v1.begin(), host_v1.end(), host_v2.begin(), real_t(0));
    const real_t host_dot = dot_block( &host_v1[ 0 ], &host_v2[ 0 ], ARRAY_SIZE, 16384);
    clock_gettime(CLOCK_MONOTONIC, &e);
    const double host_time = time_diff_ms(s, e);
    // print dot product by summing up the partially reduced vectors
    std::cout << "GPU: " << device_dot << std::endl;    

    // print dot product on cpu
    std::cout << "CPU: " << host_dot << std::endl;
    //std::cout << "CPU: " << dot( &host_v1[ 0 ], &host_v2[ 0 ], ARRAY_SIZE ) << std::endl;
    std::cout << "ELAPSED TIME(ms) kernel + cpu sum:  " << elapsed << " + " << acc << " = " << (elapsed + acc) << std::endl;
    std::cout << "TRANSFER TIME(ms):                  " << transferTime << std::endl;
    std::cout << "HOST TIME:                          " << host_time << std::endl;

    // free memory
    cudaFree( dev_v1 );
    cudaFree( dev_v2 );
    cudaFree( dev_vout );

    return 0;
}
