// #CSCS CUDA Training
//
// #Example 3.2 - transpose matrix, coalesced memory access, no bank conflicts,
//                shared memory
//
// #Author: Ugo Varetto
//

#include <iostream>
#include <vector>

typedef float real_t;

const size_t TILE_SIZE = 16;  // 16 == half warp -> coalesced access

__global__ void transpose(const real_t* in, real_t* out, int num_rows,
                          int num_columns) {
    // local cache
    __shared__ real_t tile[TILE_SIZE][TILE_SIZE];
    // locate element to transfer from input data into local cache
    // CAVEAT: size of tile == size of thread block i.e. blockDim.x ==
    // blockDim.y == TILE_SIZE
    //         this allows threads in the same warp to access data in different
    //         banks
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int input_index = row * num_columns + col;
    // 1) copy data into tile
    tile[threadIdx.y][threadIdx.x] = in[input_index];
    // wait for all threads to perform copy operation since the threads that
    // write data to the output matrix must read data which has been written
    // into cache by different threads 2) locate output element of transposed
    // matrix
    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    // transposed matrix: num_columns -> num_rows == matrix width
    const int output_index = row * num_rows + col;
    // read data of transposed element from tile
    __syncthreads();
    out[output_index] = tile[threadIdx.x][threadIdx.y];
    // note that (1) and (2) are completely separate and independent step
    // the only requirement for (2) to work is that the data are
    // available in shared memory
}

__global__ void init_matrix(real_t* in) {
    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int r = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = c + gridDim.x * blockDim.x * r;
    in[idx] = (real_t)idx;
}

void print_matrix(const real_t* m, int r, int c, int stride) {
    for (int i = 0; i != r; ++i) {
        for (int j = 0; j != c; ++j) std::cout << m[i * stride + j] << ' ';
        std::cout << '\n';
    }
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    const dim3 BLOCKS(512, 512);
    const dim3 THREADS_PER_BLOCK(16, 16);
    const int ROWS = 512 * 16;     // 8192
    const int COLUMNS = 512 * 16;  // 8192
    const size_t SIZE = ROWS * COLUMNS * sizeof(real_t);

    // device storage
    real_t* dev_in = 0;
    real_t* dev_out = 0;
    cudaMalloc(&dev_in, SIZE);
    cudaMalloc(&dev_out, SIZE);

    // host storage
    std::vector<real_t> outmatrix(ROWS * COLUMNS);

    // initialize matrix with kernel; much faster than using
    // for loops on the cpu
    init_matrix<<<dim3(COLUMNS, ROWS), 1>>>(dev_in);
    cudaMemcpy(&outmatrix[0], dev_in, SIZE, cudaMemcpyDeviceToHost);

    // print upper 4x4 left corner of input matrix
    std::cout << "INPUT MATRIX - " << ROWS << " rows, " << COLUMNS << " columns"
              << std::endl;
    print_matrix(&outmatrix[0], 4, 4, COLUMNS);

    // create events for timing execution
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop = cudaEvent_t();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record time into start event
    cudaEventRecord(start, 0);  // 0 is the default stream id
    // execute kernel
    transpose<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_in, dev_out, ROWS, COLUMNS);
    // transposeCoalesced<<<BLOCKS, THREADS_PER_BLOCK>>>>( dev_in, dev_out,
    // COLUMNS, ROWS);
    // issue request to record time into stop event
    cudaEventRecord(stop, 0);
    // synchronize stop event to wait for end of kernel execution on stream 0
    cudaEventSynchronize(stop);
    // compute elapsed time (done by CUDA run-time)
    float elapsed = 0.f;
    cudaEventElapsedTime(&elapsed, start, stop);

    std::cout << "Elapsed time (ms): " << elapsed << std::endl;

    // copy output data from device(gpu) to host(cpu)
    cudaMemcpy(&outmatrix[0], dev_out, SIZE, cudaMemcpyDeviceToHost);

    // print upper 4x4 corner of transposed matrix
    std::cout << "\nOUTPUT MATRIX - " << COLUMNS << " rows, " << ROWS
              << " columns" << std::endl;
    print_matrix(&outmatrix[0], 4, 4, ROWS);

    // free memory
    cudaFree(dev_in);
    cudaFree(dev_out);

    // release events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
