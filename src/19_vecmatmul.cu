#ifdef DOUBLE__
typedef double real_t;
#else
typedef float real_t;
#endif

//#define COLUMN //2x speed increase!

extern "C" __global__ void VecMatMul(const real_t* M,
                                     int width,
                                     int height,
                                     const real_t* V,
                                     real_t* W) {
 
#ifdef COLUMN // vector * matrix
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  //if( c >= height ) return;
  const real_t* column = M + c;
  real_t dp = 0.f;
  for( int r = 0; r < height * width; r += width )
  {
    dp += column[ r ] * V[ c ];
  }
  W[ c ] = dp;
#else // matrix * vector
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  //if( r >= width ) return;
  const real_t* row = M + r * width;
  real_t dp = 0.f;
  for( int c = 0; c != width; ++c )
  {
    dp += row[ c ] * V[ c ];
  }
  W[ r ] = dp;
#endif 
}                         
                         
                                      
  
