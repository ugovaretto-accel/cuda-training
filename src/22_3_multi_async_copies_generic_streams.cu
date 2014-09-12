//Author: Ugo Varetto
//Launch multiple asynchronous memcopies on different gpus
//and execute kernel. 
//Specify total buffer size in bytes and list of gpu ids
//on the command line.
//NOTE: the number of gpu threads used is always 1024 so the
//per-gpu buffer size (=total size / num gpus) *must* be
//evenly divisible by 1024.
//@todo automatically compute a valid thread count from
//buffer size
//
//Verify (with nvvp) that transfers happen in parallel

#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

using namespace std;

typedef signed char Int8;

#ifndef PEER_ACCESS
__global__
void Negate(Int8* buffer) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    buffer[i] = -buffer[i];
}
#else
__global__
void Negate(Int8* src, Int8* dest) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    dest[i] = -src[i];
}
#endif

void InitHostBuffer(Int8* buf, size_t hostSize, int numDevices) {
    const size_t devSize = hostSize / numDevices;
    assert(devSize);
    for(int i = 0; i != numDevices; ++i) {
        fill(buf + i * devSize, buf + i * devSize + devSize, Int8(-(i+1)));
    }
}

void EnablePeerAccess(const vector< int >& devices, int src) {
    assert(!devices.empty());
    assert(src < int(devices.size()));
    assert(src >= 0);
    int curDevice = -1;
    assert(cudaGetDevice(&curDevice) == cudaSuccess);
    assert(cudaSetDevice(devices[src]) == cudaSuccess);
    for(int i = 0; i != devices.size(); ++i) {
        const int PEER_DEVICE_TO_ACCESS = devices[i];
        const int PEER_ACCESS_FLAGS = 0;
        assert(cudaDeviceEnablePeerAccess(PEER_DEVICE_TO_ACCESS, PEER_ACCESS_FLAGS)
               == cudaSuccess); 
         
    }
    assert(cudaSetDevice(curDevice) == cudaSuccess);
}



void EnableAllToAllPeerAccess(const vector< int >& devices) {
    for(int i = 0; i != devices.size(); ++i) EnablePeerAccess(devices, i);
}


int main(int argc, char** argv) {
    assert(sizeof(Int8) == 1);
    if(argc < 2) {
        cout << "usage: " << argv[0] << " <total buffer size> <gpu ids>" << endl;
        exit(EXIT_FAILURE);
    }
    vector< int > gpus(argc - 2, -1);
    for(int i = 2; i != argc; ++i) {
        gpus[i - 2] = atoi(argv[i]);
    }
    const size_t requestedBufferSize = atoll(argv[1]);
    const int requestedNumDevices = gpus.size();
    //allocate pinned host buffer
    const size_t HOST_BUFFER_SIZE = requestedBufferSize < 1 ? 
                                    size_t(1) << 32 : requestedBufferSize;
    const int NUM_DEVICES = requestedNumDevices < 1 ? 4 : requestedNumDevices;
    const size_t DEVICE_BUFFER_SIZE = HOST_BUFFER_SIZE / NUM_DEVICES;
    assert(DEVICE_BUFFER_SIZE);
    cout << "Number of devices:      " << NUM_DEVICES << endl
         << "Buffer size:            " << HOST_BUFFER_SIZE << endl
         << "Per-device buffer size: " << DEVICE_BUFFER_SIZE << endl;
    if(HOST_BUFFER_SIZE % NUM_DEVICES != 0) {
        cout << "WARNING: buffer size NOT "
                "evenly divisible by device buffer size" << endl;
    }
    Int8* hostBuffer = 0;
    cudaError_t err = cudaMallocHost((void**) &hostBuffer, HOST_BUFFER_SIZE);
    assert(hostBuffer);
    assert(err == cudaSuccess);
    //initialize host buffer with -1-1-1-1-2-2-2-2-3-3-3-3-4-4-4-4
    InitHostBuffer(hostBuffer, HOST_BUFFER_SIZE, NUM_DEVICES);
    //allocate 4 device buffers, one per device
    vector< Int8* > deviceBuffers(NUM_DEVICES, (Int8*)(0));
    vector< cudaStream_t > streams(NUM_DEVICES, cudaStream_t());
    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        err = cudaMalloc((void**) &deviceBuffers[d], DEVICE_BUFFER_SIZE);
        assert(deviceBuffers[d]);
        assert(err == cudaSuccess);
        err = cudaStreamCreate(&streams[d]);
        assert(err == cudaSuccess);
    }
    //optioanlly enable peer access
#ifdef PEER_ACCESS
    EnableAllToAllPeerAccess(gpus.begin());
#endif     
    //async per-device copies
    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        err = cudaMemcpyAsync(deviceBuffers[d], 
                              hostBuffer + d * DEVICE_BUFFER_SIZE,
                              DEVICE_BUFFER_SIZE, 
                              cudaMemcpyHostToDevice,
                              streams[d]);
        assert(err == cudaSuccess);
    }
#ifdef PEER_ACCESS    
    cudaDeviceSynchronize();
#endif
    const int THREAD_BLOCK_SIZE = 1024;
    const int BLOCK_SIZE = DEVICE_BUFFER_SIZE / THREAD_BLOCK_SIZE;
    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        const bool KERNEL_ENABLED_OPTION = d == 0; //only enable for first device
#ifdef PEER_ACCESS
        Negate<<< BLOCK_SIZE, THREAD_BLOCK_SIZE, 0, streams[d] >>>(
            deviceBuffers[(d + 1) % NUM_DEVICES], deviceBuffers[d]);
#else
        Negate<<< BLOCK_SIZE, THREAD_BLOCK_SIZE, 0, streams[d] >>>(deviceBuffers[d]);
#endif        
#ifdef CHECK_KERNEL_LAUNCH       
        err == cudaGetLastError(); //no idea about what this does, does it trigger a barrier ?
        assert(err == cudaSuccess);
#endif
    }
    //
    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        err = cudaMemcpyAsync(hostBuffer + d * DEVICE_BUFFER_SIZE,
                              deviceBuffers[d], 
                              DEVICE_BUFFER_SIZE,
                              cudaMemcpyDeviceToHost,
                              streams[d]);
        assert(err == cudaSuccess);
    }

    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess);
    }
#ifdef PEER_ACCESS
    for(int d = 0; d != NUM_DEVICES; ++d) {
        for(Int8* p = hostBuffer + d * DEVICE_BUFFER_SIZE;
            p != hostBuffer + d * DEVICE_BUFFER_SIZE + DEVICE_BUFFER_SIZE;
            ++p);// assert(*p == 1);
    }
#else
    for(int d = 0; d != NUM_DEVICES; ++d) {
        for(Int8* p = hostBuffer + d * DEVICE_BUFFER_SIZE;
            p != hostBuffer + d * DEVICE_BUFFER_SIZE + DEVICE_BUFFER_SIZE;
            ++p) assert(*p == (d + 1));
    }
#endif    
    err = cudaFreeHost(hostBuffer);
    assert(err == cudaSuccess);
    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        err = cudaFree(deviceBuffers[d]);
        err = cudaStreamDestroy(streams[d]);
        assert(err == cudaSuccess);
    }
    err = cudaDeviceReset();
    assert(err == cudaSuccess);
    cout << "PASSED" << endl;
    return 0;
}
