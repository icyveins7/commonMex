#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>

#define THREADS_PER_BLOCK 128
#define NUM_GPUS 2

struct t_data{
	int thread_t_ID;
	
	Npp16sc *thread_h_indata;
	Npp32f *thread_h_ftap;
	
	
	int thread_datalen;
	int thread_L;
	int thread_N;
	int thread_Dec;
	
	
	Npp32fc *thread_h_outdata;
};

// declare global thread stuff
struct t_data t_data_array[NUM_GPUS];

//// === Cuda kernels ===
// let's try with block-wise per nprimePt, thread-wise per fft point, so that maybe we can improve occupancy
// no occupancy change, but slightly faster?
__global__
void wola_front_sm_tuned(int N, int L, int Dec, int nprimePts, Npp16sc *d_in, Npp32fc *d_out, Npp32f *d_ftapg)
{
	extern __shared__ float s[];
	Npp32f *d_ftap = s;

	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x; 
	
	for (int l = threadIdx.x; l < L; l += blockDim.x){ 
		d_ftap[l] = d_ftapg[l];
	}
	
	// wait for copies to shared mem to end
	__syncthreads();
	
	
	Npp32f re, im; // oh snap writing to stack first almost halves the kernel time lol
	int n;
	
	
	for (int nprime = blockIdx.x; nprime < nprimePts; nprime+=gridDim.x){
		n = nprime*Dec;
		for (int a = threadIdx.x; a<N; a+=blockDim.x){ 
			re = 0;
			im = 0;
			
			for (int b = 0; b < L/N; b++){
				if (n - (b*N+a) >= 0){
					re = re + (Npp32f)(d_in[n - (b*N+a)].re) * d_ftap[b*N+a];
					im = im + (Npp32f)(d_in[n - (b*N+a)].im) * d_ftap[b*N+a];
				}
			}
			
			d_out[nprime*N + a].re = re;
			d_out[nprime*N + a].im = im;
		}
		
	}
	
}

unsigned __stdcall gpuWola_Xt(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	Npp16sc *h_indata = inner_data->thread_h_indata;
	Npp32f *h_ftap = inner_data->thread_h_ftap;
	
	int datalen = inner_data->thread_datalen;
	int L = inner_data->thread_L;
	int N = inner_data->thread_N;
	int Dec = inner_data->thread_Dec;
	
	Npp32fc *h_outdata = inner_data->thread_h_outdata;
	// end of attached variables

	// // GPU setting variables
	// int device;
	// cudaDeviceProp prop;
	
	// set the GPU for this thread
	cudaSetDevice(t_ID);
	// cudaGetDevice(&device);
	// cudaGetDeviceProperties(&prop, device);
	// printf("Allocating memory for GPU Device: %s\n", prop.name);
	
	// device memory
	int nprimePts = datalen/Dec;
	Npp16sc *d_indata;
	Npp32f *d_ftap;
	Npp32fc *d_outdata;
	cudaMalloc((void**)&d_indata, sizeof(Npp16sc)*datalen);
	cudaMalloc((void**)&d_ftap, sizeof(Npp32f)*L);
	cudaMalloc((void**)&d_outdata, sizeof(Npp32fc)*nprimePts*N);
	
	// cufft plan
	cufftHandle batchplan;
	
	int fftlen[1] = {N};
	int istride = 1;
	int inembed[1] = {N * nprimePts};
	int idist = N;
	int ostride = 1;
	int onembed[1] = {N * nprimePts};
	int odist = N;
	cufftResult cfr = cufftPlanMany(&batchplan,1,fftlen,
				   inembed,istride,idist,
				   onembed,ostride,odist,
				   CUFFT_C2C,nprimePts);
	if (cfr != CUFFT_SUCCESS){mexErrMsgTxt("cufft Plan failed!\n");}
	
	
	// copy the data into device
	cudaMemcpy(d_indata,h_indata,datalen*sizeof(Npp16sc), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ftap,h_ftap,L*sizeof(Npp32f), cudaMemcpyHostToDevice);
	
	// run the kernel
	wola_front_sm_tuned<<<nprimePts, THREADS_PER_BLOCK, sizeof(Npp32f) * L >>>(N, L, Dec, nprimePts, d_indata, d_outdata, d_ftap);
	
	// run the ffts
	cufftExecC2C(batchplan, (cufftComplex*)d_outdata, (cufftComplex*)d_outdata, CUFFT_INVERSE); // in-place
	
	// copy output back
	cudaMemcpy(h_outdata, d_outdata, N*nprimePts*sizeof(Npp32fc), cudaMemcpyDeviceToHost);
	
	// cleanup gpu 
	cudaFree(d_indata);
	cudaFree(d_ftap);
	cudaFree(d_outdata);
	
	cufftDestroy(batchplan);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
// calling arguments, out = (signal, f_tap, N, Dec)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// // GPU setting variables
	// int device;
	// cudaDeviceProp prop;
	
	
    // declare variables
    Npp16sc *h_indata, *h_indata2;
	// Npp16sc *d_indata, *d_indata2;
	
	float *h_ftap;
	// Npp32f *d_ftap, *d_ftap2;
	
	int datalen, datalen2; // length of indata
	int L; // length of filter
	int N; // number of channels
	int Dec; // decimation factor
	
	// declare outputs
	int nprimePts[2];
	// Npp32fc *d_outdata, *d_outdata2;
	mxComplexSingle *h_outdata, *h_outdata2;
	

	//reserve stuff for threads
    int t; 
    HANDLE ThreadList[NUM_GPUS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 Inputs required.");
    }

	// get inputs and check
	h_indata = (Npp16sc*)mxGetComplexInt16s(prhs[0]);
	datalen = (int)mxGetM(prhs[0]) * (int)mxGetN(prhs[0]);
	
	h_ftap = (float*)mxGetSingles(prhs[1]);
	L = (int)mxGetM(prhs[1]) * (int)mxGetN(prhs[1]);
	
	if (L > 8192){
        mexErrMsgTxt("ERROR: Length of filter only supported up to 8192!");
    }
	
	N = (int)mxGetScalar(prhs[2]);
	if (L%N != 0){
        mexErrMsgTxt("ERROR: Filter taps length must be factor multiple of fft length!");
    }
	
	Dec = (int)mxGetScalar(prhs[3]);
	// if (Dec != N){
		// mexErrMsgTxt("ERROR: Decimation must be equal to number of channels! Other decimation ratios not yet implemented!");
	// }
	
	// get the next input and check too
	h_indata2 = (Npp16sc*)mxGetComplexInt16s(prhs[4]);
	if (!mxIsComplex(prhs[4]) || !mxIsInt16(prhs[4])){
		mexErrMsgTxt("ERROR: Both inputs must be complex int16s!");
	}
	
	datalen2 = (int)mxGetM(prhs[4]) * (int)mxGetN(prhs[4]);
	
	
	/* create the output matrix */
	if (datalen%Dec!=0){
		mexErrMsgTxt("ERROR: Input data must be a multiple of supplied decimation factor!");
	}
	nprimePts[0] = datalen/Dec;
	nprimePts[1] = datalen2/Dec;
    plhs[0] = mxCreateNumericMatrix(N,nprimePts[0], mxSINGLE_CLASS,mxCOMPLEX);
	plhs[1] = mxCreateNumericMatrix(N,nprimePts[1], mxSINGLE_CLASS,mxCOMPLEX);
	h_outdata = mxGetComplexSingles(plhs[0]);
	h_outdata2 = mxGetComplexSingles(plhs[1]);
    
    for (t=0; t<NUM_GPUS; t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_h_ftap = (Npp32f*)h_ftap;
		
		t_data_array[t].thread_L = L;
		t_data_array[t].thread_N = N;
		t_data_array[t].thread_Dec = Dec;
		
		// GPU-specific allocations (lazy to make a pointer of pointers)
		if (t==0){	
			t_data_array[t].thread_h_indata = h_indata;
			
			t_data_array[t].thread_datalen = datalen;
			
			t_data_array[t].thread_h_outdata = (Npp32fc*)h_outdata;
		}
		if (t==1){
			t_data_array[t].thread_h_indata = h_indata2;
			
			t_data_array[t].thread_datalen = datalen2;
			
			t_data_array[t].thread_h_outdata = (Npp32fc*)h_outdata2;
		}
		
		// start the thread
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&gpuWola_Xt,(void*)&t_data_array[t],0,NULL);
	}
	
	WaitForMultipleObjects(NUM_GPUS,ThreadList,1,INFINITE);
	
	// ============== CLEANUP =================
    // close threads
    for(t=0;t<NUM_GPUS;t++){
        CloseHandle(ThreadList[t]);
    }
	
}