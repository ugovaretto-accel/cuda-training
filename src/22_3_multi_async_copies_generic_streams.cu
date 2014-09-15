//Author: Ugo Varetto
//Launch multiple asynchronous memcopies on different gpus
//and execute kernel. 
//Specify total buffer size in bytes and list of gpu ids
//on the command line.
//Compile with -DPEER_ACCESS to enable peer access
//NOTE: the number of gpu threads used is always 1024 so the
//per-gpu buffer size (=total size / num gpus) *must* be
//evenly divisible by 1024.
//@todo automatically compute a valid thread count from
//buffer size
//
//Verify (with nvvp) that transfers happen in parallel
//In case PEER_ACCESS is enabled:
// - data is initialized as usual
// - kernels running on evenly indexed GPUs access memory 
//   from oddly indexed GPUs to updated a memory buffer
//   allocated on the same GPUs where the kernel is running


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
//return index of buffer to read data from from current GPU
int MapKey(int current, int numdevices) {
    return (current + 1) % numdevices;
}
__host__ __device__
int Op(Int8 a, Int8 b) { return a - b; }
//dest: gpu executing the code
//src: remote gpu
__global__
void Map(Int8* src, Int8* dest) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    dest[i] = Op(dest[i], src[i]);
}
//check that value va was properly computed
bool Check(Int8 val, Int8 i, int numdevices) {
    return val == Op(Int8(-(i + 1)), Int8(-(MapKey(i, numdevices) + 1)));
}
#endif

//initialze host buffer: buffer 'i' is inititalized with '-(i+1)'
void InitHostBuffer(Int8* buf, size_t hostSize, int numDevices) {
    const size_t devSize = hostSize / numDevices;
    assert(devSize);
    for(int i = 0; i != numDevices; ++i) {
        fill(buf + i * devSize, buf + i * devSize + devSize, Int8(-(i+1)));
    }
}

//Enable device wth index 'src' to access all the other devices
void EnablePeerAccess(const vector< int >& devices, int src) {
    assert(!devices.empty());
    assert(src < int(devices.size()));
    assert(src >= 0);
    int curDevice = -1;
    assert(cudaGetDevice(&curDevice) == cudaSuccess);
    assert(cudaSetDevice(devices[src]) == cudaSuccess);
    for(int i = 0; i != devices.size(); ++i) {
        const int PEER_DEVICE_TO_ACCESS = devices[i];
        if(PEER_DEVICE_TO_ACCESS == devices[src]) continue;
        const int PEER_ACCESS_FLAGS = 0;
        cout << "Enabling access to: " << PEER_DEVICE_TO_ACCESS
             << " from " << src << endl;
        assert(cudaDeviceEnablePeerAccess(PEER_DEVICE_TO_ACCESS, PEER_ACCESS_FLAGS)
               == cudaSuccess); 
         
    }
    assert(cudaSetDevice(curDevice) == cudaSuccess);
}

//Enable peer access using MapKey function to determine
//which device must evenly indexed gpus access
void EnableMappedPeerAccess(const vector< int >& devices) {
    int curDevice = -1;
    assert(cudaGetDevice(&curDevice) == cudaSuccess);
    for(int i = 0; i < devices.size(); i += 2) {
        assert(cudaSetDevice(devices[i]) == cudaSuccess);
        assert(cudaDeviceEnablePeerAccess(devices[MapKey(i, int(devices.size()))], 0)
            == cudaSuccess);
    }
    assert(cudaSetDevice(curDevice) == cudaSuccess);
}

//main
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
    const size_t HOST_BUFFER_SIZE = requestedBufferSize < 1 ? 
                                    size_t(1) << 32 : requestedBufferSize;
    const int NUM_DEVICES = requestedNumDevices < 1 ? 4 : requestedNumDevices;
    const size_t DEVICE_BUFFER_SIZE = HOST_BUFFER_SIZE / NUM_DEVICES;
    assert(DEVICE_BUFFER_SIZE);
    int numdevices = -1;
    cudaError_t err = cudaGetDeviceCount(&numdevices);
    assert(err == cudaSuccess);
    assert(int(gpus.size()) <= numdevices && numdevices > 0);
    //check that there are no duplicate elements in the input indices
    assert(unique(gpus.begin(), gpus.end()) == gpus.end());
    //check that each element is >=0 & < gpus.size())
    assert(*min_element(gpus.begin(), gpus.end()) >= 0);
    assert(*max_element(gpus.begin(), gpus.end()) < numdevices);
    cout << "Number of devices:      " << NUM_DEVICES << endl
         << "Buffer size:            " << HOST_BUFFER_SIZE << endl
         << "Per-device buffer size: " << DEVICE_BUFFER_SIZE << endl;
    if(HOST_BUFFER_SIZE % NUM_DEVICES != 0) {
        cout << "WARNING: buffer size NOT "
                "evenly divisible by device buffer size" << endl;
    }
    Int8* hostBuffer = 0;
    //allocate pinned host buffer
    err = cudaMallocHost((void**) &hostBuffer, HOST_BUFFER_SIZE);
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
    //optionally enable peer access
#ifdef PEER_ACCESS
    EnableMappedPeerAccess(gpus);
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
    //temporary: replace with proper event-based synchronization;
    //since each gpu only needs to wait on another gpu use
    //events to sync streams
    for(int d = 0; d != gpus.size(); ++d) {
        err = cudaSetDevice(gpus[d]);
        assert(err == cudaSuccess);
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess);
    }
#endif
    const int THREAD_BLOCK_SIZE = 1024;
    const int BLOCK_SIZE = DEVICE_BUFFER_SIZE / THREAD_BLOCK_SIZE;
    for(int d = 0; d != NUM_DEVICES; ++d) {
        const int gpu = gpus[d];
        err = cudaSetDevice(gpu);
        assert(err == cudaSuccess);
        const bool KERNEL_ENABLED_OPTION = d == 0; //only enable for first device
#ifdef PEER_ACCESS
        if(d % 2 != 0) continue; //even ids read from odd ids
        Map<<< BLOCK_SIZE, THREAD_BLOCK_SIZE, 0, streams[d] >>>(
                    deviceBuffers[MapKey(d,NUM_DEVICES)], deviceBuffers[d]);
        cout << gpus[MapKey(d,NUM_DEVICES)] << "->" << gpus[d] << endl;
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
        if(d % 2 != 0) continue; //even ids read from odd ids
        const int src = (d + 1) % NUM_DEVICES;
        for(Int8* p = hostBuffer + d * DEVICE_BUFFER_SIZE;
            p != hostBuffer + d * DEVICE_BUFFER_SIZE + DEVICE_BUFFER_SIZE;
            ++p)if(!Check(*p, d, NUM_DEVICES)) {
                cout << d << ' ' << int(*p) << endl; return 1;//assert(Check(*p, d, NUM_DEVICES));
}
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
