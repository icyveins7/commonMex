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

// timing functions
double PCFreq = 0.0;
__int64 CounterStart = 0;
int StartCounter()
{
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
    printf("QueryPerformanceFrequency failed!\n");

    PCFreq = ((double)li.QuadPart)/1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
	return (int)CounterStart;
}

int GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (int)li.QuadPart;
}

//// === Cuda kernels ===
// let's try with block-wise per nprimePt, thread-wise per fft point, so that maybe we can improve occupancy
// no occupancy change, but slightly faster?
// the d_in_buffer is now Dec*nprimePts + L length ie cyclelen + L
__global__
void wola_front_sm_tuned_delay(int N, int L, int Dec, int nprimePts, Npp16sc *d_in_buffer, Npp32fc *d_out, Npp32f *d_ftapg)
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
	// Npp16sc *d_in = &d_in_buffer[L];
	
	
	for (int nprime = blockIdx.x; nprime < nprimePts; nprime+=gridDim.x){
		n = nprime*Dec;
		for (int a = threadIdx.x; a<N; a+=blockDim.x){ 
			re = 0;
			im = 0;
			
			for (int b = 0; b < L/N; b++){
				// if (n - (b*N+a) >= 0){
					re = re + (Npp32f)(d_in_buffer[L + n - (b*N+a)].re) * d_ftap[b*N+a];
					im = im + (Npp32f)(d_in_buffer[L + n - (b*N+a)].im) * d_ftap[b*N+a];
				// }
			}
			
			// if(re == 0 && im == 0){printf("%i, %i, %g %g\n", blockIdx.x, threadIdx.x, re, im);}
			
			d_out[nprime*N + a].re = re;
			d_out[nprime*N + a].im = im;
		}
		
	}
	
}


/* The gateway function */
// calling arguments, out = (signal, f_tap, N, Dec)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// Timing
	int start_t = StartCounter();
	int end_t;
	
    // declare variables
    Npp16sc *h_indata;
	Npp16sc *h_indata_matlab; // non-pinned
	Npp16sc *d_indata_buffer0, *d_indata_buffer1;
	
	float *h_ftap;
	Npp32f *d_ftap;
	
	int datalen; // length of indata
	int L; // length of filter
	int N; // number of channels
	int Dec; // decimation factor
	
	// declare outputs
	int nprimePts;
	int cycles;
	Npp32fc *d_outdata_buffer0, *d_outdata_buffer1;
	Npp32fc *h_outdata;
	mxComplexSingle *h_outdata_matlab; // non-pinned
	
	

	// //reserve stuff for threads
    // int t; 
    // HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 Inputs required.");
    }

	// get inputs and check
	h_indata_matlab = (Npp16sc*)mxGetComplexInt16s(prhs[0]);
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
	if (Dec != N){
		mexErrMsgTxt("ERROR: Decimation must be equal to number of channels! Other decimation ratios not yet implemented!");
	}
	
	cycles = (int)mxGetScalar(prhs[4]);
	if (datalen % cycles != 0){
		mexErrMsgTxt("ERROR: Cycles must be a factor of input data length!");
	}
	int cyclelen = datalen/cycles;
	
	
	/* create the output matrix */
	if (datalen%Dec!=0){
		mexErrMsgTxt("ERROR: Input data must be a multiple of supplied decimation factor!");
	}
	nprimePts = datalen/Dec;
    plhs[0] = mxCreateNumericMatrix(N,nprimePts, mxSINGLE_CLASS,mxCOMPLEX);
	h_outdata_matlab = mxGetComplexSingles(plhs[0]);
	
	int outlen = nprimePts * N;
	int cycle_nprimePts = nprimePts/cycles;
	int cycleOutlen = outlen/cycles;
    
    // allocate device memory
	cudaMalloc((void**)&d_ftap, sizeof(Npp32f)*L);
	cudaMalloc((void**)&d_indata_buffer0, sizeof(Npp16sc)*(cyclelen+L)); // allocate more for the delay, in a continguous fashion!
	cudaMalloc((void**)&d_indata_buffer1, sizeof(Npp16sc)*(cyclelen+L)); // allocate the waiting buffer
	cudaMalloc((void**)&d_outdata_buffer0, sizeof(Npp32fc)*cycleOutlen);
	cudaMalloc((void**)&d_outdata_buffer1, sizeof(Npp32fc)*cycleOutlen);
	
	// allocate pinned host memory for async transfers
	cudaMallocHost((void**)&h_indata, sizeof(Npp16sc)*datalen);
	cudaMallocHost((void**)&h_outdata, sizeof(Npp32fc)*outlen);
	
	// initialize input data by memcpy
	start_t = GetCounter();
	memcpy(h_indata, h_indata_matlab, sizeof(Npp16sc) * datalen);
	end_t = GetCounter();
	printf("Copying input to pinned memory, took %g ms.\n", (end_t - start_t)/PCFreq);
	
	// cufft parameters, batchmode
	cufftHandle batchplan;
	int fftlen[1] = {N};
	int istride = 1;
	int inembed[1] = {N * cycle_nprimePts};
	int idist = N;
	int ostride = 1;
	int onembed[1] = {N * cycle_nprimePts};
	int odist = N;
	cufftResult cfr = cufftPlanMany(&batchplan,1,fftlen,
				   inembed,istride,idist,
				   onembed,ostride,odist,
				   CUFFT_C2C,cycle_nprimePts);
	if (cfr != CUFFT_SUCCESS){mexErrMsgTxt("cufft Plan failed!\n");}
	
	// make a stream for compute and for copying to enable async stuff (seems like having this number of streams is the only way to ensure the overlapped transfers..)
	cudaStream_t computeStream;
	cudaStream_t copyStreamHtoD;
	cudaStream_t copyStreamDtoH;
	cudaStream_t copyStreamDtoD;
	cudaStreamCreate(&computeStream);
	cudaStreamCreate(&copyStreamHtoD);
	cudaStreamCreate(&copyStreamDtoD);
	cudaStreamCreate(&copyStreamDtoH);
	cufftSetStream(batchplan, computeStream);
	
	// copy the f_tap
	cudaMemcpy(d_ftap,h_ftap,L*sizeof(Npp32f), cudaMemcpyHostToDevice);
	
	// make sure we zero the first buffer's delay
	nppSetStream(computeStream);
	nppsZero_16sc(d_indata_buffer0, L + cyclelen);
	
	// issue the first copies!
	cudaMemcpy(&d_indata_buffer0[L], h_indata, cyclelen*sizeof(Npp16sc), cudaMemcpyHostToDevice); // use non-async?
	
	// assign pointers for clarity?
	Npp16sc *d_indata_current = d_indata_buffer0;
	Npp16sc *d_indata_waiting = d_indata_buffer1;
	Npp16sc *d_indata_swap; // used to swap them later
	Npp32fc *d_outdata_current = d_outdata_buffer0;
	Npp32fc *d_outdata_copying = d_outdata_buffer1;
	Npp32fc *d_outdata_swap;
	
	// ==start the computational loop==
	int i;
	for (i = 0; i < cycles-1; i++){ // only cycle to -1 of the total, otherwise you will copy out of bounds
	// for (i=0;i<1;i++){
		// wait for previous async transfers of the next batch data to end 
		cudaStreamSynchronize(copyStreamHtoD);
		cudaStreamSynchronize(copyStreamDtoD);
		

		// copy the end of the first buffer into the delay of the waiting buffer
		cudaMemcpyAsync(&d_indata_waiting[0], &d_indata_current[cyclelen], L*sizeof(Npp16sc), cudaMemcpyDeviceToDevice, copyStreamDtoD); 
		cudaStreamQuery(copyStreamDtoD);
		
		// copy the new cycle's data after that into the rest of the waiting buffer
		cudaMemcpyAsync(&d_indata_waiting[L], &h_indata[(i+1)*cyclelen], (cyclelen)*sizeof(Npp16sc), cudaMemcpyHostToDevice, copyStreamHtoD);
		cudaStreamQuery(copyStreamHtoD);
		
		// run the kernel on the computeStream
		
		wola_front_sm_tuned_delay<<<cycle_nprimePts, THREADS_PER_BLOCK, sizeof(Npp32f) * L , computeStream>>>(N, L, Dec, cycle_nprimePts, d_indata_current, d_outdata_current, d_ftap);
		cudaStreamQuery(computeStream);
		
		// for the consequent ones, copy the pinned to non-pinned (EMBED THIS AT THIS POSITION IN THE CODE BECAUSE THE WOLA_FRONT IS THE LONGEST CALL AND CPU MEMCPY IS BLOCKING)
		if (i>0){
			memcpy(&h_outdata_matlab[(i-1)*cycleOutlen], &h_outdata[(i-1)*cycleOutlen], sizeof(Npp32fc)*cycleOutlen);
		}
		
		
		// wait for compute stream to finish then do the ffts
		// cudaStreamSynchronize(computeStream);
		cfr = cufftExecC2C(batchplan, (cufftComplex*)d_outdata_current, (cufftComplex*)d_outdata_current, CUFFT_INVERSE); // try in-place?
		if(cfr!=CUFFT_SUCCESS){printf("ERROR WHILE COMPUTING CUFFT ON ITERATION %i\n", i);}
	
		// then wait for the compute stream to finish
		cudaStreamSynchronize(computeStream);
		
		// swap pointers and copy the output back to host
		cudaStreamSynchronize(copyStreamDtoH);
		
		
		d_outdata_swap = d_outdata_current;
		d_outdata_current = d_outdata_copying; // at this point the d_outdata_current is ready for writing to already
		d_outdata_copying = d_outdata_swap;
		cudaMemcpyAsync(&h_outdata[i*cycleOutlen], d_outdata_copying, cycleOutlen*sizeof(Npp32fc), cudaMemcpyDeviceToHost, copyStreamDtoH);
		cudaStreamQuery(copyStreamDtoH);
		
		// swap pointers
		d_indata_swap = d_indata_current;
		d_indata_current = d_indata_waiting; // so waiting -> current
		d_indata_waiting = d_indata_swap; // and current -> waiting
		
		
	}
	
	// wait for data to transfer in
	cudaStreamSynchronize(copyStreamHtoD);
	cudaStreamSynchronize(copyStreamDtoD);
	
	// run the kernel on the computeStream
	wola_front_sm_tuned_delay<<<cycle_nprimePts, THREADS_PER_BLOCK, sizeof(Npp32f) * L , computeStream>>>(N, L, Dec, cycle_nprimePts, d_indata_current, d_outdata_current, d_ftap);
	cudaStreamQuery(computeStream);
	
	// again, embed the memcpy here
	memcpy(&h_outdata_matlab[(cycles-2)*cycleOutlen], &h_outdata[(cycles-2)*cycleOutlen], sizeof(Npp32fc)*cycleOutlen);
	
	
	// wait for compute stream to finish then do the ffts
	// cudaStreamSynchronize(computeStream);
	cfr = cufftExecC2C(batchplan, (cufftComplex*)d_outdata_current, (cufftComplex*)d_outdata_current, CUFFT_INVERSE); // try in-place?
	if(cfr!=CUFFT_SUCCESS){printf("ERROR WHILE COMPUTING CUFFT ON ITERATION %i\n", i);}

	// then wait for the compute stream to finish
	cudaStreamSynchronize(computeStream);
	
	// swap pointers and copy the output back to host
	cudaStreamSynchronize(copyStreamDtoH);
	d_outdata_swap = d_outdata_current;
	d_outdata_current = d_outdata_copying; // at this point the d_outdata_current is ready for writing to already
	d_outdata_copying = d_outdata_swap;
	cudaMemcpyAsync(&h_outdata[(cycles-1)*cycleOutlen], d_outdata_copying, cycleOutlen*sizeof(Npp32fc), cudaMemcpyDeviceToHost, copyStreamDtoH);
	cudaStreamQuery(copyStreamDtoH);
	
	// wait for all of it to finish
	cudaDeviceSynchronize();
	
	// final memcpy for the final batch
	memcpy(&h_outdata_matlab[(cycles-1)*cycleOutlen], &h_outdata[(cycles-1)*cycleOutlen], sizeof(Npp32fc)*cycleOutlen);
	
	// // copy output to matlab array
	// start_t = GetCounter();
	// memcpy(h_outdata_matlab, h_outdata, sizeof(Npp32fc)*outlen);
	// end_t = GetCounter();
	// printf("Copying output from pinned mem to matlab array, took %g ms.\n", (end_t - start_t)/PCFreq);
	
	// ==end of computational loop==
	
	// cleanup
	cudaFree(d_indata_buffer0); cudaFree(d_indata_buffer1);
	cudaFree(d_ftap);
	cudaFree(d_outdata_buffer0); cudaFree(d_outdata_buffer1);
	
	cudaFreeHost(h_indata); cudaFreeHost(h_outdata);
	
	cufftDestroy(batchplan);
	
	cudaStreamDestroy(copyStreamHtoD);
	cudaStreamDestroy(copyStreamDtoH);
	cudaStreamDestroy(copyStreamDtoD);
	cudaStreamDestroy(computeStream);
	
}