// #CSCS CUDA Training 
//
// #Exercise 11 - surfaces
//
// #Author Ugo Varetto
//
// #Goal: compare the performance of 2D stencil application with:
//        1) global memory
//        2) texture memory
//        3) shared memory
//        4) surfaces   
//
// #Rationale: shows how texture memory is faster than global memory
//             when data are reused, thanks to (2D) caching; also
//             shows that for periodic boundary conditions using hw wrapping
//             is much faster than performing manual bounds checking
//
// #Solution: implement stencil computation accessing data in global, texture and shared memory
//
// #Code: 1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) map texture memory to pre-allocated gpu storage
//        4) copy data from host to device
//        5) launch kernel
//        6) read data back
//        7) consume data (in this case print result)
//        8) release texture memory 
//        9) free memory
//        
// #Compilation: [no wrap] nvcc -arch=sm_13 11_surfaces.cu -o surfaces
//               [wrap   ] nvcc -DTEXTURE_WRAP -arch=sm_13 11_surfaces.cu -o surfaces
//
// #Execution: ./surfaces
//
// #warning: texture wrap mode doesn't seem to work with non-power-of-two textures 
//
// #Note: textures do not support directly 64 bit (double precision) floating point data 
//        it is however possible to unpack doubles into int2 textures and reconstruct the double inside
//        a kernel local variable
//
// #Note: Global time / Cached time == Cached time / Texture time ~= 2
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better



//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>


//------------------------------------------------------------------------------
// read input data from global memory
__global__ void apply_stencil( const float* gridIn, 
                               const float* stencil,
                               float* gridOut,
                               int gridNumRows,
                               int gridNumColumns,
                               int stencilSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int soff = halfStencilSize;
    float s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i ) {
        si = gridI + i;
        if( si < 0 ) si += gridNumRows;
        else if( si >= gridNumRows ) si -= gridNumRows;
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
            sj = gridJ + j;
            if( sj < 0 ) sj += gridNumColumns;
            else if( sj >= gridNumColumns ) sj -= gridNumColumns;
            s += gridIn[ si * gridNumColumns + sj ] * 
                 stencil[ ( i + soff ) * stencilSize + ( j + soff ) ];
        }
    }
    gridOut[ gridI * gridNumColumns + gridJ ] = s;
}




//------------------------------------------------------------------------------
// texture references wrapping global memory
texture< float, 2 > gridInTex;
texture< float, 2 > stencilTex;


// read input data from global memory
__global__ void apply_stencil_texture( float* gridOut,
                                       int gridNumRows,
                                       int gridNumColumns,
                                       int stencilSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int soff = halfStencilSize;
    float s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i ) {
        si = gridI + i;
#ifndef TEXTURE_WRAP
        if( si < 0 ) si += gridNumRows;
        else if( si >= gridNumRows ) si -= gridNumRows;
#endif              
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
             sj = gridJ + j;
#ifndef TEXTURE_WRAP
             if( sj < 0 ) sj += gridNumColumns;
             else if( sj >= gridNumColumns ) sj -= gridNumColumns;
#endif                               
             s += tex2D( gridInTex, sj, si ) * 
                  tex2D( stencilTex, j + soff, i + soff );
        }
    }
    gridOut[ gridI * gridNumColumns + gridJ ] = s;
}

//------------------------------------------------------------------------------
// texture references wrapping array
texture< float, 2 > gridInTexArray;
texture< float, 2 > stencilTexArray;


// read input data from global memory
__global__ void apply_stencil_texture_array( float* gridOut,
                                             int gridNumRows,
                                             int gridNumColumns,
                                             int stencilSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int soff = halfStencilSize;
    float s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i ) {
        si = gridI + i;
#ifndef TEXTURE_WRAP
        if( si < 0 ) si += gridNumRows;
        else if( si >= gridNumRows ) si -= gridNumRows;
#endif              
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
             sj = gridJ + j;
#ifndef TEXTURE_WRAP
             if( sj < 0 ) sj += gridNumColumns;
             else if( sj >= gridNumColumns ) sj -= gridNumColumns;
#endif                               
             s += tex2D( gridInTexArray, sj, si ) * 
                  tex2D( stencilTexArray, j + soff, i + soff );
        }
    }
    gridOut[ gridI * gridNumColumns + gridJ ] = s;
}

//------------------------------------------------------------------------------
// texture references wrapping array
surface< void, 2 > gridOutSurf; // <- can write!


// read input data from global memory
__global__ void apply_stencil_surface( int gridNumRows,
                                       int gridNumColumns,
                                       int stencilSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    const int xStride =  sizeof( float ); 
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int soff = halfStencilSize;
    float s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i ) {
        si = gridI + i;
#ifndef TEXTURE_WRAP
        if( si < 0 ) si += gridNumRows;
        else if( si >= gridNumRows ) si -= gridNumRows;
#endif              
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
             sj = gridJ + j;
#ifndef TEXTURE_WRAP
             if( sj < 0 ) sj += gridNumColumns;
             else if( sj >= gridNumColumns ) sj -= gridNumColumns;
#endif                               
             s += tex2D( gridInTexArray, sj, si ) * 
                  tex2D( stencilTexArray, j + soff, i + soff );
             // to read from surfaces: 
             //surf2Dread( &s, ... );
        }
    }
    surf2Dwrite( s,  gridOutSurf, xStride * gridJ, gridI, cudaBoundaryModeTrap );
}


//------------------------------------------------------------------------------
// read input data from global memory, cache block into local(shared) memory

__device__ float get_grid_element( const float* grid,
                                    int row, 
                                    int column, 
                                    int numRows,
                                    int numColumns ) {                             
    if( row < 0 ) row += numRows;
    else if( row >= numRows ) row -= numRows;
    if( column < 0 ) column += numColumns;
    else if( column >= numColumns ) column -= numColumns;                                   
    return  grid[ row * numColumns + column ];
}

// threads + half stencil edge X threads + half stencil edge + stencil buffer size
extern __shared__ float cache[];


__global__ void apply_stencil_cached( const float* gridIn, 
                                      const float* stencilIn,
                                      float* gridOut,
                                      int gridNumRows,
                                      int gridNumColumns,
                                      int stencilSize,
                                      int tileNumRows,
                                      int tileNumColumns ) {

    float* localGrid = &cache[ 0 ];
    float* stencil   = &cache[ 0 ] + tileNumRows * tileNumColumns;
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    
    if( threadIdx.x < stencilSize && threadIdx.y < stencilSize ) {
        const int si = threadIdx.y * stencilSize + threadIdx.x;
        stencil[ si ] = stencilIn[ si ];
    }
       
    // 1) copy into shared memory; shared memory is 
    //    ( blockDim.x + halfStencilSize x 2 ) x ( blockDim.x + halfStencilSize x 2 )
    // move to upper left corner of local grid
    const int row = threadIdx.y + halfStencilSize;
    const int col = threadIdx.x + halfStencilSize;
    // if corner copy data into corners of halo region
    if( ( threadIdx.x < halfStencilSize || threadIdx.x >= blockDim.x - halfStencilSize ) &&
        ( threadIdx.y < halfStencilSize || threadIdx.y >= blockDim.y - halfStencilSize ) ) {
    
        int coff = 0;
        int roff = 0;
        if( threadIdx.y < halfStencilSize ) roff = -halfStencilSize;
        else if( threadIdx.y >= blockDim.y - halfStencilSize ) roff = halfStencilSize;
        if( threadIdx.x < halfStencilSize ) coff = -halfStencilSize;
        else if( threadIdx.x >= blockDim.x - halfStencilSize ) coff = halfStencilSize;
        localGrid[ ( row + roff ) * tileNumColumns + ( col + coff )  ] = 
            get_grid_element( gridIn, gridI + roff, gridJ + coff, gridNumRows, gridNumColumns );
           
    }
    // copy element from grid
    localGrid[ row * tileNumColumns + col ] =
        get_grid_element( gridIn,  gridI, gridJ, gridNumRows, gridNumColumns );
    // if row < half stencil edge size also copy element into upper and lower sides of halo region
    if( threadIdx.y < halfStencilSize ) {
        localGrid[ ( row - halfStencilSize ) * tileNumColumns + col ] =
            get_grid_element( gridIn,  gridI - halfStencilSize, gridJ, gridNumRows,
                              gridNumColumns );
        localGrid[ ( row + blockDim.y ) * tileNumColumns + col ] =
            get_grid_element( gridIn,  gridI + blockDim.y, gridJ, gridNumRows,
                              gridNumColumns );             
    }
    // if column < half stencil edge size also copy element into left and right sides of halo region
    if( threadIdx.x < halfStencilSize ) {
        localGrid[ row * tileNumColumns + col - halfStencilSize ] =
            get_grid_element( gridIn,  gridI, gridJ - halfStencilSize, gridNumRows,
                              gridNumColumns );
        localGrid[ row * tileNumColumns + col + blockDim.x ] =
            get_grid_element( gridIn,  gridI, gridJ + blockDim.x, gridNumRows, gridNumColumns );            
    }
    // wait until local cache is filled
    __syncthreads();
    // apply stencil to local (cached) grid               
    const int soff = halfStencilSize;
    float s = 0.f; 
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i ) {
        const int si = row + i;
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
            const int sj = col + j;
            s += localGrid[ si * tileNumColumns + sj ] * stencil[ (i+soff)*stencilSize + (j+soff)];
        }
    }
    // write result into output grid
    gridOut[ gridI * gridNumColumns + gridJ ] = s;
}


//------------------------------------------------------------------------------
void apply_stencil_ref( const float* gridIn,
                        const float* stencil,
                        float* gridOut,
                        int gridNumRows,
                        int gridNumColumns,
                        int stencilSize ) {
                                                    
     const int halfStencilSize = stencilSize / 2;
     const int soff = halfStencilSize;
     for( int r = 0; r != gridNumRows; ++r ) {
         for( int c = 0; c != gridNumColumns; ++c ) {
             float s = 0.f; 
             int si = 0;
             int sj = 0;
             for( int i = -halfStencilSize; i <= halfStencilSize; ++i ) {
                 si = r + i;
                 if( si < 0 ) si += gridNumRows;
                 else if( si >= gridNumRows ) si -= gridNumRows;
                 for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
                      sj = c + j;
                      if( sj < 0 ) sj += gridNumColumns;
                      else if( sj >= gridNumColumns ) sj -= gridNumColumns;
                     s += gridIn[ si * gridNumColumns + sj ] *
                          stencil[ ( i + soff ) * stencilSize + ( j + soff ) ];
                 }
             }     
             gridOut[ r * gridNumColumns + c ] = s;
         }
     }
}

__global__ void init_grid( float* grid ) {
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    grid[ gridI * stride + gridJ ] = float( ( gridI + gridJ ) % 2 );                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
}

//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int GRID_NUM_ROWS    = 0x800;// + 1; //257
    const int GRID_NUM_COLUMNS = 0x800;// + 1; //257
    const int GRID_SIZE = GRID_NUM_ROWS * GRID_NUM_COLUMNS;
    const int GRID_BYTE_SIZE = sizeof( float ) * GRID_SIZE;
    const int DEVICE_BLOCK_NUM_ROWS = 16; // num threads per row
    const int DEVICE_BLOCK_NUM_COLUMNS = 16; // num threads per columns
    const int STENCIL_EDGE_LENGTH = 3;
    const int STENCIL_SIZE = STENCIL_EDGE_LENGTH * STENCIL_EDGE_LENGTH;
    const int STENCIL_BYTE_SIZE = sizeof( float ) * STENCIL_SIZE;
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int DEVICE_GRID_NUM_ROWS    = 
        ( GRID_NUM_ROWS    + DEVICE_BLOCK_NUM_ROWS    - 1 ) / DEVICE_BLOCK_NUM_ROWS;
    const int DEVICE_GRID_NUM_COLUMNS = 
        ( GRID_NUM_COLUMNS + DEVICE_BLOCK_NUM_COLUMNS - 1 ) / DEVICE_BLOCK_NUM_COLUMNS;
    // if number of threads is not evenly divisable by the number of threads per block 
    // we need an additional block; the above code can be rewritten as
    // if( NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) 
    //     BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK;
    // else BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1 
 
    //host allocated storage
    std::vector< float > host_stencil( STENCIL_SIZE, 1.0f / STENCIL_SIZE );
    std::vector< float > host_grid_in( GRID_SIZE );
    std::vector< float > host_grid_out( GRID_SIZE );

    // gpu allocated storage
    float* dev_grid_in  = 0;
    float* dev_grid_out = 0;
    float* dev_stencil  = 0;
    cudaMalloc( &dev_grid_in,  GRID_BYTE_SIZE    );
    cudaMalloc( &dev_grid_out, GRID_BYTE_SIZE   );
    cudaMalloc( &dev_stencil,  STENCIL_BYTE_SIZE );
 
    // copy stencil to device
    cudaMemcpy( dev_stencil, &host_stencil[ 0 ], STENCIL_BYTE_SIZE, cudaMemcpyHostToDevice );

    init_grid<<< dim3( GRID_NUM_ROWS, GRID_NUM_COLUMNS, 1), dim3( 1, 1, 1 ) >>>( dev_grid_in );

    // copy initialized grid to host grid, faster than initializing on CPU
    cudaMemcpy( &host_grid_in[ 0 ], dev_grid_in, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );

    const dim3 blocks( DEVICE_GRID_NUM_COLUMNS, DEVICE_GRID_NUM_ROWS, 1 );
    const dim3 threads_per_block( DEVICE_BLOCK_NUM_COLUMNS, DEVICE_BLOCK_NUM_ROWS, 1 ); 

    //--------------------------------------------------------------------------
    // initialize events for timing execution
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    float e = 0.f;

    cudaEventRecord( start );
    
    // execute kernel accessing global memory
    apply_stencil<<<blocks, threads_per_block>>>( dev_grid_in,
                                                  dev_stencil,
                                                  dev_grid_out,
                                                  GRID_NUM_ROWS,
                                                  GRID_NUM_COLUMNS,
                                                  STENCIL_EDGE_LENGTH );
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &host_grid_out[ 0 ], dev_grid_out, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    // print grid
    std::cout << "Global memory - result:  " << host_grid_out.front() << ".." 
              << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms\n" << std::endl; 

    //--------------------------------------------------------------------------
    // describe data inside texture: 1-component floating point value in this case    
    const int BITS_PER_BYTE = 8;
    cudaChannelFormatDesc cd = cudaCreateChannelDesc( sizeof( float ) *  BITS_PER_BYTE,
                                                      0, 0, 0, cudaChannelFormatKindFloat );
#ifdef TEXTURE_WRAP    
    gridInTex.addressMode[ 0 ] = cudaAddressModeWrap;
    gridInTex.addressMode[ 1 ] = cudaAddressModeWrap;
#endif                                                      
    // bind textures to pre-allocated storage
    int texturePitch = sizeof( float ) * GRID_NUM_COLUMNS;
    cudaBindTexture2D( 0, &gridInTex,   dev_grid_in, &cd, GRID_NUM_COLUMNS,
                       GRID_NUM_ROWS, texturePitch );
    texturePitch = sizeof( float ) * STENCIL_EDGE_LENGTH;
    cudaBindTexture2D( 0, &stencilTex,  dev_stencil, &cd, STENCIL_EDGE_LENGTH,
                       STENCIL_EDGE_LENGTH, texturePitch );                                                  

    cudaEventRecord( start );

    // execute kernel accessing global memory
    apply_stencil_texture<<<blocks, threads_per_block>>>( dev_grid_out,
                                                          GRID_NUM_ROWS,
                                                          GRID_NUM_COLUMNS,
                                                          STENCIL_EDGE_LENGTH );
    
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // release texture
    cudaUnbindTexture( &gridInTex  );
    cudaUnbindTexture( &stencilTex );
    // read back result
    cudaMemcpy( &host_grid_out[ 0 ], dev_grid_out, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    // print grid
    std::cout << "Texture memory - result:  " << host_grid_out.front() << ".." 
              << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms\n" << std::endl;
    
    //--------------------------------------------------------------------------  
#ifdef TEXTURE_WRAP    
    gridInTexArray.addressMode[ 0 ] = cudaAddressModeWrap;
    gridInTexArray.addressMode[ 1 ] = cudaAddressModeWrap;
#endif

    cudaArray* dev_grid_in_array = 0;
    cudaArray* dev_stencil_array = 0;
    cudaMallocArray( &dev_grid_in_array, &cd, GRID_NUM_COLUMNS, GRID_NUM_ROWS );
    cudaMallocArray( &dev_stencil_array, &cd, STENCIL_EDGE_LENGTH, STENCIL_EDGE_LENGTH );
    cudaMemcpyToArray( dev_grid_in_array, 0, 0, dev_grid_in, GRID_BYTE_SIZE,    
                       cudaMemcpyDeviceToDevice );
    cudaMemcpyToArray( dev_stencil_array, 0, 0, dev_stencil, STENCIL_BYTE_SIZE,
                       cudaMemcpyDeviceToDevice );
                                                         
    // bind textures to array
    cudaBindTextureToArray( &gridInTexArray,  dev_grid_in_array, &cd );
    cudaBindTextureToArray( &stencilTexArray, dev_stencil_array, &cd );                                                  

    cudaEventRecord( start );

    // execute kernel accessing global memory
    apply_stencil_texture_array<<<blocks, threads_per_block>>>( dev_grid_out,
                                                                GRID_NUM_ROWS,
                                                                GRID_NUM_COLUMNS,
                                                                STENCIL_EDGE_LENGTH );
    
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
   
    // read back result
    cudaMemcpy( &host_grid_out[ 0 ], dev_grid_out, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    // print grid
    std::cout << "Texture arrays - result: " << host_grid_out.front() << ".." 
              << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms\n" << std::endl;

   //--------------------------------------------------------------------------  
    cudaArray* dev_grid_out_array = 0;
    cudaMallocArray( &dev_grid_out_array, &cd, GRID_NUM_COLUMNS, GRID_NUM_ROWS,
                     cudaArraySurfaceLoadStore ); // <- ALLOW WRITE OPERATIONS!                                                   
                                          

    // bind surface to array
    cudaBindSurfaceToArray( gridOutSurf, dev_grid_out_array );    
 
    cudaEventRecord( start );

    // execute kernel accessing global memory
    apply_stencil_surface<<<blocks, threads_per_block>>>( GRID_NUM_ROWS,
                                                          GRID_NUM_COLUMNS,
                                                          STENCIL_EDGE_LENGTH );
    
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
   
    // read back result
    cudaMemcpyFromArray( &host_grid_out[ 0 ], dev_grid_out_array, 0, 0, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    cudaThreadSynchronize();

    // print grid
    std::cout << "Texture -> Surface- result: " << host_grid_out.front() << ".." 
              << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms\n" << std::endl;    

    //--------------------------------------------------------------------------
    // initialize events for timing execution
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    cudaEventRecord( start );
    
    // / -> int div:  2 * ( I / 2 ) != I when I odd
    const int TILE_NUM_ROWS    = threads_per_block.x + 2 * ( STENCIL_EDGE_LENGTH / 2 );
    const int TILE_NUM_COLUMNS = TILE_NUM_ROWS; 
    const int SHARED_MEM_SIZE  = sizeof( float ) * TILE_NUM_ROWS * TILE_NUM_COLUMNS +
                                 STENCIL_BYTE_SIZE;

    // execute kernel accessing global memory
    apply_stencil_cached<<< blocks, threads_per_block, SHARED_MEM_SIZE >>>( dev_grid_in,
                                                                            dev_stencil,
                                                                            dev_grid_out,
                                                                            GRID_NUM_ROWS,
                                                                            GRID_NUM_COLUMNS,
                                                                            STENCIL_EDGE_LENGTH,
                                                                            TILE_NUM_ROWS,
                                                                            TILE_NUM_COLUMNS );
  
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &host_grid_out[ 0 ], dev_grid_out, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    // print grid
    std::cout << "Shared memory - result:  " << host_grid_out.front() << ".." 
              << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms\n" << std::endl; 

    //--------------------------------------------------------------------------
    apply_stencil_ref( &host_grid_in[ 0 ],
                       &host_stencil[ 0 ],
                       &host_grid_out[ 0 ],
                       GRID_NUM_ROWS,
                       GRID_NUM_COLUMNS,
                       STENCIL_EDGE_LENGTH );
    std::cout << "CPU - result:            " << host_grid_out.front() << ".." 
              << host_grid_out.back() << std::endl;

    // release texture
    cudaUnbindTexture( &gridInTex  );
    cudaUnbindTexture( &stencilTex );

    // release arrays
    cudaFreeArray( dev_grid_in_array );
    cudaFreeArray( dev_stencil_array );

    // free memory
    cudaFree( dev_grid_in );
    cudaFree( dev_grid_out );
    cudaFree( dev_stencil );

    return 0;
}
