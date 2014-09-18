//Author: Ugo Varetto
//Launch multiple asynchronous memcopies on different gpu streams
//and execute kernel. 
//Specify total buffer size and number of streams
//on the command line.
//NOTE: the number of gpu threads used is always 1024 so the
//per-stream buffer size (=total size / num streams) *must* be
//evenly divisible by 1024.
//@todo automatically compute a valid thread count from
//buffer size
//
//Verify (with nvvp) that transfers happen in parallel

#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef signed char Int8;

__global__
void Negate(Int8* buffer) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    buffer[i] = -buffer[i];
}


void InitHostBuffer(Int8* buf, size_t hostSize, int numDevices) {
    const size_t devSize = hostSize / numDevices;
    assert(devSize);
    for(int i = 0; i != numDevices; ++i) {
        fill(buf + i * devSize, buf + i * devSize + devSize, Int8(-(i+1)));
    }
}


int main(int argc, char** argv) {
    assert(sizeof(Int8) == 1);
    if(argc < 2) {
        cout << "usage: " << argv[0] << " <total buffer size> <number of streams>" << endl;
    }
    
    const size_t requestedBufferSize = atoll(argv[1]);
    const int requestedNumStreams = atoi(argv[2]);
    //allocate pinned host buffer
    const size_t HOST_BUFFER_SIZE = requestedBufferSize < 1 ? 
                                    size_t(1) << 32 : requestedBufferSize;
    const int NUM_STREAMS = requestedNumStreams < 1 ? 4 : requestedNumStreams;
    const size_t STREAM_BUFFER_SIZE = HOST_BUFFER_SIZE / NUM_STREAMS;
    assert(STREAM_BUFFER_SIZE);
    cout << "Number of streams:      " << NUM_STREAMS           << endl
         << "Buffer size:            " << HOST_BUFFER_SIZE      << endl
         << "Per-stream buffer size:  " << STREAM_BUFFER_SIZE    << endl;
    if(HOST_BUFFER_SIZE % NUM_STREAMS != 0) {
        cout << "WARNING: buffer size NOT "
                "evenly divisible by stram buffer size" << endl;
    }
    Int8* hostBuffer = 0;
    cudaError_t err = cudaMallocHost((void**) &hostBuffer, HOST_BUFFER_SIZE);
    assert(hostBuffer);
    assert(err == cudaSuccess);
    //initialize host buffer with -1-1-1-1-2-2-2-2-3-3-3-3-4-4-4-4
    InitHostBuffer(hostBuffer, HOST_BUFFER_SIZE, NUM_STREAMS);
    //allocate 4 device buffers, one per device
    vector< Int8* > streamBuffers(NUM_STREAMS, (Int8*)(0));
    vector< cudaStream_t > streams(NUM_STREAMS, cudaStream_t());
    for(int d = 0; d != NUM_STREAMS; ++d) {
        err = cudaMalloc((void**) &streamBuffers[d], STREAM_BUFFER_SIZE);
        assert(streamBuffers[d]);
        assert(err == cudaSuccess);
        err = cudaStreamCreate(&streams[d]);
        assert(err == cudaSuccess);
    }
    //async per-device copies
#ifdef SEPARATE_COPY_EXECUTE
    for(int d = 0; d != NUM_STREAMS; ++d) {
        err = cudaMemcpyAsync(streamBuffers[d], 
                              hostBuffer + d * STREAM_BUFFER_SIZE,
                              STREAM_BUFFER_SIZE, cudaMemcpyHostToDevice,
                              streams[d]);
        assert(err == cudaSuccess);
    }
#endif
    //
    const int THREAD_BLOCK_SIZE = 1024;
    const int BLOCK_SIZE = STREAM_BUFFER_SIZE / THREAD_BLOCK_SIZE;
    for(int d = 0; d != NUM_STREAMS; ++d) {
#ifndef SEPARATE_COPY_EXECUTE        
        err = cudaMemcpyAsync(streamBuffers[d], 
                              hostBuffer + d * STREAM_BUFFER_SIZE,
                              STREAM_BUFFER_SIZE, cudaMemcpyHostToDevice,
                              streams[d]);
        assert(err == cudaSuccess);
#endif        
        Negate<<< BLOCK_SIZE, THREAD_BLOCK_SIZE, 0, streams[d] >>>(streamBuffers[d]);
#ifdef CHECK_KERNEL_LAUNCH       
        err == cudaGetLastError(); //no idea about what this does, does it trigger a barrier ?
        assert(err == cudaSuccess);
#endif
#ifndef SEPARATE_COPY_EXECUTE        
        err = cudaMemcpyAsync(hostBuffer + d * STREAM_BUFFER_SIZE,
                              streamBuffers[d], 
                              STREAM_BUFFER_SIZE, cudaMemcpyDeviceToHost,
                              streams[d]);
        assert(err == cudaSuccess);
#endif        
    }
#ifdef SEPARATE_COPY_EXECUTE
    for(int d = 0; d != NUM_STREAMS; ++d) {
        err = cudaMemcpyAsync(hostBuffer + d * STREAM_BUFFER_SIZE,
                              streamBuffers[d], 
                              STREAM_BUFFER_SIZE, cudaMemcpyDeviceToHost,
                              streams[d]);
        assert(err == cudaSuccess);
    }
#endif    

   // for(int d = 0; d != NUM_STREAMS; ++d) {
   //     err = cudaStreamSynchronize(streams[d]);
   //     assert(err == cudaSuccess);
   // }
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    for(int d = 0; d != NUM_STREAMS; ++d) {
        for(Int8* p = hostBuffer + d * STREAM_BUFFER_SIZE;
            p != hostBuffer + d * STREAM_BUFFER_SIZE + STREAM_BUFFER_SIZE;
            ++p) assert(*p == (d + 1));
    }

    err = cudaFreeHost(hostBuffer);
    assert(err == cudaSuccess);
    for(int d = 0; d != NUM_STREAMS; ++d) {
        err = cudaFree(streamBuffers[d]);
        assert(err == cudaSuccess);
        err = cudaStreamDestroy(streams[d]);
        assert(err == cudaSuccess);
    }
    err = cudaDeviceReset();
    assert(err == cudaSuccess);
    cout << "PASSED" << endl;
    return 0;
}
