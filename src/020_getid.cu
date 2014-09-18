
__device__ int getid() { 
  return blockIdx.x * blockDim.x + threadIdx.x;
}
