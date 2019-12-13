// -lnppc_static -lnpps_static -lculibos -lcufft_static

#include "mex.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#define LIGHT_SPD 299792458.0

// defines for lla2ecef
#define A_SQ 4.068063159076900E+13
#define B_SQ 4.040829998466145E+13
#define M_PI 3.14159265358979323846

// // Convenience function for checking CUDA runtime API results
// // can be wrapped around any runtime API call. No-op in release builds.
// inline
// cudaError_t checkCuda(cudaError_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
  // if (result != cudaSuccess) {
    // fprintf(stderr, "CUDA Runtime Error: %sn", 
            // cudaGetErrorString(result));
    // assert(result == cudaSuccess);
  // }
// #endif
  // return result;
// }

__device__
void d_lla2ecefDeg(float latitude, float longitude, float ellipsoid_height, float *ecef)
{
	// convert to rads
	float lat  = latitude * M_PI / 180;   
	float lon = longitude * M_PI / 180;  
	float cos_lat = cos(lat);  
	float sin_lat = sin(lat);
	float cos_lon = cos(lon); 
	float sin_lon = sin(lon);

	float N  = A_SQ / sqrt( A_SQ*cos_lat*cos_lat + B_SQ*sin_lat*sin_lat);

	ecef[0]  = (N + ellipsoid_height) * cos_lat * cos_lon;
	ecef[1]  = (N + ellipsoid_height) * cos_lat * sin_lon;
	ecef[2]  = (B_SQ*N/A_SQ + ellipsoid_height) * sin_lat;
	
}

__global__
void grid_search_kernel_float(float *latlist, int latpts, float *lonlist, int lonpts, float *tdoa_g, float *sigma_g, int num_tdoas, float *sensor_pos_g, int num_sens, int *pairs_g, float *cost)
{
	extern __shared__ float s[];
	float *sensor_pos = s; // num_sens * 3 
	float *tdoa = (float*)&sensor_pos[num_sens*3]; // num_tdoas
	float *sigma = (float*)&tdoa[num_tdoas]; // num_tdoas
	int *pairs = (int*)&sigma[num_tdoas]; // num_tdoas * 2
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	// copy the sensor_pos, tdoas, sigmas, and pairs into the shared memory
	for (int k = threadIdx.x; k < num_sens * 3; k += blockDim.x){
		sensor_pos[k] = sensor_pos_g[k];
	}
	
	for (int j = threadIdx.x; j < num_tdoas; j += blockDim.x){ // just gonna reuse from thread 0, i don't think it's too important?
		tdoa[j] = tdoa_g[j];
		sigma[j] = sigma_g[j];
	}
	
	for (int l = threadIdx.x; l < num_tdoas * 2; l += blockDim.x){ // just gonna reuse from thread 0, i don't think it's too important?
		pairs[l] = pairs_g[l];
	}
	
	// wait for copies to shared mem to end
	__syncthreads();

	
	// now calculate using shared mem
	float temp_val; // declare one on stack to calculate within before writing to global mem
	float theo_tdoa;

	float t0[3]; // float vectors on stack for calculations
	float t1[3];
	int idx0, idx1;
	// float fma_res0, fma_res1;
	float grid_pos[3];
	float lat, lon;
	
	for (int i = index; i < latpts;  i += stride){ // iterate over latpts
		for (int j = 0; j < lonpts; j++){
			lat = latlist[i];
			lon = lonlist[i];
			
			temp_val = 0;
		
			// calculate theoretical tdoa and compare
			for (int t = 0; t < num_tdoas; t++){
				idx0 = pairs[t * 2 + 0];
				idx1 = pairs[t * 2 + 1];
				
				for (int a = 0; a < 3; a++){
					t0[a] = sensor_pos[idx0*3 + a] - grid_pos[i*3 + a];
					t1[a] = sensor_pos[idx1*3 + a] - grid_pos[i*3 + a];
				}
				
				
				
				// // use fma for better accuracy?
				// fma_res0 = t0[0] * t0[0];
				// fma_res1 = t1[0] * t1[0];
				// fma_res0 = fmaf(t0[1],t0[1],fma_res0);
				// fma_res0 = fmaf(t0[2],t0[2],fma_res0);
				// fma_res1 = fmaf(t1[1],t1[1],fma_res1);
				// fma_res1 = fmaf(t1[2],t1[2],fma_res1);
				
				// // theo_tdoa =  (sqrtf(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]) - sqrt(t0[0]*t0[0] + t0[1]*t0[1] + t0[2]*t0[2]) ) / LIGHT_SPD;
				// theo_tdoa = (sqrtf(fma_res1) - sqrtf(fma_res0)) / LIGHT_SPD;
				
				// or directly use norms?
				theo_tdoa = ( normf(3, t1) - normf(3, t0) ) / LIGHT_SPD;
				
				temp_val = temp_val + (theo_tdoa - tdoa[t]) * (theo_tdoa - tdoa[t]) / sigma[t] / sigma[t];
			}
			
			// now write the output to the global mem
			cost[i] = temp_val;
	}
	
	
}


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
    size_t free_byte ;
	size_t total_byte ;
  
	cudaMemGetInfo( &free_byte, &total_byte );
	printf("Before alloc, GPU memory usage: free = %f MB, total = %f MB\n",(double)free_byte/1024.0/1024.0,(double)total_byte/1024.0/1024.0);
	
	// declare vars
    float *h_gridpts, *h_sigma, *h_tdoas, *h_sens_pos, *h_cost;
	float *d_gridpts, *d_sigma, *d_tdoas, *d_sens_pos, *d_cost;
	int *h_pairs, *d_pairs;
	
	int num_gridpts;
	int num_sens;
	int num_tdoas;
	 
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 Inputs required.");
    }

	h_gridpts = (float*)mxGetSingles(prhs[0]);
    h_sigma = (float*)mxGetSingles(prhs[1]); 
	h_tdoas = (float*)mxGetSingles(prhs[2]);
	h_sens_pos = (float*)mxGetSingles(prhs[3]);
	h_pairs = (int*)mxGetInt32s(prhs[4]);
    
	num_gridpts = (int)mxGetN(prhs[0]);
	num_sens = (int)mxGetN(prhs[1]);
	num_tdoas = (int)(mxGetM(prhs[2]) * mxGetN(prhs[2]));
	printf("num_gridpts = %i, num_sens = %i, num_tdoas = %i \n", num_gridpts, num_sens, num_tdoas);

	//// allocate memory on device
	// checkCuda(cudaMallocHost((void**)&h_gridpts, sizeof(float)*num_gridpts*3));
	// checkCuda(cudaMallocHost((void**)&h_sigma, sizeof(float)*num_tdoas));
	// checkCuda(cudaMallocHost((void**)&h_tdoas, sizeof(float)*num_tdoas));
	// checkCuda(cudaMallocHost((void**)&h_sens_pos, sizeof(float)*num_sens*3));
	// checkCuda(cudaMallocHost((void**)&h_cost, sizeof(float)*num_gridpts));
	// checkCuda(cudaMallocHost((void**)&h_pairs, sizeof(int)*num_tdoas*2));
	
	cudaMalloc((void**)&d_gridpts, sizeof(float)*num_gridpts*3);
	cudaMalloc((void**)&d_sigma, sizeof(float)*num_tdoas);
	cudaMalloc((void**)&d_tdoas, sizeof(float)*num_tdoas);
	cudaMalloc((void**)&d_sens_pos, sizeof(float)*num_sens*3);
	cudaMalloc((void**)&d_cost, sizeof(float)*num_gridpts);
	cudaMalloc((void**)&d_pairs, sizeof(int)*num_tdoas*2);
	
	cudaMemGetInfo( &free_byte, &total_byte );
	printf("After alloc, GPU memory usage: free = %f MB, total = %f MB\n",(double)free_byte/1024.0/1024.0,(double)total_byte/1024.0/1024.0);
	
	

	
	/* create the output matrix */
	plhs[0] = mxCreateUninitNumericMatrix(1,num_gridpts,mxSINGLE_CLASS,mxREAL);
    /* get a pointer to the real data in the output matrix */
	h_cost = (float*)mxGetSingles(plhs[0]);

	// transfer data to the gpu
	cudaMemcpy(d_gridpts,h_gridpts,num_gridpts*3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tdoas,h_tdoas,num_tdoas*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sigma,h_sigma,num_tdoas*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sens_pos,h_sens_pos,num_sens*3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairs,h_pairs,num_tdoas*2*sizeof(int), cudaMemcpyHostToDevice);
	
	// run the kernel
	grid_search_kernel_float<<<1, 1024, (num_sens*3 + num_tdoas * 2)*sizeof(float) + (num_tdoas*2) * sizeof(int)>>>(d_gridpts, num_gridpts, d_tdoas, d_sigma, num_tdoas, d_sens_pos, num_sens, d_pairs, d_cost);
	
	// copy the costs back
	cudaMemcpy(h_cost, d_cost, num_gridpts*sizeof(float), cudaMemcpyDeviceToHost);

	

    // cleanup
	// cudaFreeHost(h_gridpts);
	// cudaFreeHost(h_tdoas);
	// cudaFreeHost(h_sigma);
	// cudaFreeHost(h_sens_pos);
	// cudaFreeHost(h_cost);
	// cudaFreeHost(h_pairs);
	
	cudaFree(d_gridpts);
	cudaFree(d_tdoas);
	cudaFree(d_sigma);
	cudaFree(d_sens_pos);
	cudaFree(d_cost);
	cudaFree(d_pairs);
   
   // cudaDeviceReset();
   // mexAtExit(cleanUp);
}
