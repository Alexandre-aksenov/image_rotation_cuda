/*
* This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Timer.h"
#include "Util.h"

#define DIMX 512
#define DIMY 512
char *image_filename 	= "lena_bw.raw";
char *res_image			= "lena_bw_rot.raw";
char *ref_filename   	= "ref_rotated.raw";
float angle = 1.f;    // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 1

// declare texture reference for 2D float texture

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__global__ void
transformKernel( float* g_odata, cudaTextureObject_t texObj, int width, int height, float theta) 
{
    // calculate normalized texture coordinates; coord o`u le code va 'ecrire
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x / (float) width;
    float v = y / (float) height;

    // transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u*cosf(theta) - v*sinf(theta) + 0.5f;
    float tv = v*cosf(theta) + u*sinf(theta) + 0.5f;   

    // read from texture and write to global memory
    g_odata[y*width + x] = tex2D<float>(texObj, tu, tv);   //tex est l'im de depart, en coords normalisees,  
    /* Bilinear  interpolation to determine the
    "optimal pixel" at the given position. */  
    
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int
main( int argc, char** argv) 
{

	Timer t1;
	int width = DIMX;
	int height = DIMY;

	// Init the device
    cudaDeviceProp deviceProp;
    int devID = 0;
    SafeCall(cudaSetDevice(devID));
    SafeCall(cudaGetDeviceProperties(&deviceProp, devID));
    printf("> Using CUDA device [%d]: %s - cap: %d.%d\n", devID, deviceProp.name,deviceProp.major,deviceProp.minor);

    // load image from disk
    float* h_data = NULL;		// host pointer
	char * image_filename="lena_bw_rot.pgm";
	load_image(image_filename,&h_data,width,height);

    unsigned char* h_data_byte_ref = NULL;
    image_filename="ref_rotated.raw";
	load_image_c(image_filename,&h_data_byte_ref,width,height);

	unsigned int size_f = DIMX*DIMY*sizeof(float);

    // allocate device memory for result
    float* d_data = NULL;
    SafeCall( cudaMalloc( (void**) &d_data, size_f));

    // allocate array and copy image data				     <- float
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    SafeCall( cudaMallocArray( &cu_array, &channelDesc, width, height )); /*SafeCall (?)  */
    SafeCall( cudaMemcpyToArray( cu_array, 0, 0, h_data, size_f, cudaMemcpyHostToDevice));
    
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    SafeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    dim3 dimBlock(16, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	t1.start();
    // execute the kernel
    transformKernel<<< dimGrid, dimBlock, 0 >>>( d_data, texObj, width, height, angle);
    // check if kernel execution generated an error
    LastError("Kernel execution failed");

    cudaDeviceSynchronize();
	t1.stop();

    printf("Processing time: %f (ms)\n", t1.getElapsedTimeInMilliSec());
    printf("%.2f Mpixels/sec\n", (width*height / (t1.getElapsedTimeInMilliSec()/ 1000.0f)) / 1e6);


    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( size_f);
    // copy result from device to host
    SafeCall( cudaMemcpy( h_odata, d_data, size_f, cudaMemcpyDeviceToHost));
    
    // write result to file

	char * res_image = "results/lena_bw_rot.pgm";
    write_image(res_image,h_odata,DIMX,DIMY);

    SafeCall(cudaDestroyTextureObject(texObj));
    SafeCall(cudaFree(d_data));
    SafeCall(cudaFreeArray(cu_array));
    free(h_data);
    free(h_odata);

    cudaDeviceReset();
    return 0;
}
