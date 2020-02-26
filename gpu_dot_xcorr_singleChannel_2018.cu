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
#define MAX_GPU_MEMORY_ALLOC 8589934592 // 8 GB

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

// struct t_data{
	// int thread_t_ID;
	
	// mxComplexSingle *thread_x;
	// mxComplexSingle *thread_y;
	// int *thread_startIdx_list;
	// int thread_xcorrIterTotal;
	// int *thread_shiftsZero;
	// int thread_shiftPts;
	// int thread_mullen;
	// int thread_nfft;
	
	// mxComplexSingle *thread_ww;
	// mxComplexSingle *thread_aa;
	// mxComplexSingle *thread_fv;
	// int thread_k_bins;
	
	// // IPP DFT vars
	// Ipp8u *thread_pBuffer;
	// IppsDFTSpec_C_32fc *thread_pSpec;
	
	// float *thread_dualqf2;
	// int *thread_dual_shiftsInd;
	// int *thread_dual_freqInd;
// };

// // declare global thread stuff
// struct t_data t_data_array[NUM_THREADS];

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

// the most naive kernel possible
__global__
void power_spectr_kernel(int len, Npp32fc *d_in, Npp32f *d_powerspectr)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = index; i < len; i = i + stride){
		d_powerspectr[i] = d_in[i].re * d_in[i].re + d_in[i].im * d_in[i].im;
	}
	
}

// unsigned __stdcall xcorr_dualChannel(void *pArgs){
	// struct t_data *inner_data;
	// inner_data = (struct t_data *)pArgs;
	
	// int t_ID = inner_data->thread_t_ID;
	
	// mxComplexSingle *x = inner_data->thread_x;
	// mxComplexSingle *y = inner_data->thread_y;
	// int *startIdx_list = inner_data->thread_startIdx_list; // list of startIdx to go to
	// int xcorrIterTotal = inner_data->thread_xcorrIterTotal;
	// int *shiftsZero = inner_data->thread_shiftsZero;
	// int shiftPts = inner_data->thread_shiftPts;
	// int mullen = inner_data->thread_mullen; // technically no longer the actual fftlen, but rather the multiplylen
	// int nfft = inner_data->thread_nfft;
	
	// mxComplexSingle *ww = inner_data->thread_ww;
	// mxComplexSingle *aa = inner_data->thread_aa;
	// mxComplexSingle *fv = inner_data->thread_fv;
	// int k_bins = inner_data->thread_k_bins;
	
	// // IPP DFT vars
	// Ipp8u *pBuffer = inner_data->thread_pBuffer;
	// IppsDFTSpec_C_32fc *pSpec = inner_data->thread_pSpec;
	
	// float *dualqf2 = inner_data->thread_dualqf2;
	// int *dual_shiftsInd = inner_data->thread_dual_shiftsInd;
	// int *dual_freqInd = inner_data->thread_dual_freqInd;
	// // end of attached variables

	// // computations
	// int i, xcorrIter, curr_shift;
	// int startIdx;
	// float cutout_pwr;
	// double cutout_pwr_64f;
	// float y_pwr;
	// double y_pwr_64f;
	// Ipp32s *shifts = (Ipp32s*)ippsMalloc_32s_L(shiftPts);
	// Ipp32fc *cutout = (Ipp32fc*)ippsMalloc_32fc_L(mullen);
	// Ipp32fc *dft_in = (Ipp32fc*)ippsMalloc_32fc_L(nfft);
	// Ipp32fc *dft_out = (Ipp32fc*)ippsMalloc_32fc_L(nfft);
	// Ipp32fc *conv_out = (Ipp32fc*)ippsMalloc_32fc_L(nfft);
	// Ipp32f *magnSq = (Ipp32f*)ippsMalloc_32f_L(k_bins);
	// Ipp32f maxval;
	// int maxind;
	
	// // temp arrays for single xcorr equivalent
	// Ipp32f *productpeaks = (Ipp32f*)ippsMalloc_32f_L(shiftPts);
	// Ipp32s *freqlist_inds = (Ipp32s*)ippsMalloc_32s_L(shiftPts);
	
	// // zero out the dft_in so the padded part is all zeroes
	// ippsZero_32fc(dft_in, nfft);
	
	// // pick xcorrIter based on thread number
	// for (xcorrIter = t_ID; xcorrIter<xcorrIterTotal; xcorrIter = xcorrIter + NUM_THREADS){
		// startIdx = startIdx_list[xcorrIter];
		// ippsAddC_32s_Sfs((Ipp32s*)shiftsZero, (Ipp32s)startIdx, shifts, shiftPts, 0); // scale factor of 0 implies the same value
		// ippsConj_32fc((Ipp32fc*)&x[startIdx-1],cutout, mullen); // need to save the conjugate in cutout, need -1 to convert from matlab indexing
		// ippsNorm_L2_32fc64f((Ipp32fc*)&x[startIdx-1], mullen, &cutout_pwr_64f);
		// cutout_pwr = (float)(cutout_pwr_64f*cutout_pwr_64f);
		
		// for (i = 0; i<shiftPts; i++){
			// curr_shift = shifts[i]-1;
			// // printf("Working on thread %i, loop %i, shift %i \n",thread,i,curr_shift);
			
			// ippsNorm_L2_32fc64f((Ipp32fc*)&y[curr_shift], mullen, &y_pwr_64f);
			// y_pwr = (float)(y_pwr_64f*y_pwr_64f);
			
			// ippsMul_32fc((Ipp32fc*)cutout,(Ipp32fc*)&y[curr_shift], (Ipp32fc*)dft_in, mullen); // so we multiply the short mullen
			
			// // now do the convolution!
			// ippsMul_32fc_I((Ipp32fc*)aa, dft_in, mullen);
			
			// ippsDFTFwd_CToC_32fc(dft_in, dft_out, pSpec, pBuffer); // but we dft the longer nfft
			
			// ippsMul_32fc_I((Ipp32fc*)fv, dft_out, nfft);
			
			// ippsDFTInv_CToC_32fc(dft_out, conv_out, pSpec, pBuffer);
			
			// ippsMul_32fc(&conv_out[mullen-1], (Ipp32fc*)&ww[mullen-1], dft_out, k_bins); // we reuse the start of dft_out to store the shorter k_bins
			
			// ippsPowerSpectr_32fc(dft_out, magnSq, k_bins); // we calculate magnSq of the start of dft_out, only k_bins length
			
			// ippsMaxIndx_32f(magnSq, k_bins, &maxval, &maxind);
			
			// productpeaks[i] = maxval/cutout_pwr/y_pwr;
			// freqlist_inds[i] = maxind;
		// }
		
		// // now find the max in the single productpeaks
		// ippsMaxIndx_32f(productpeaks, shiftPts, &maxval, &maxind);
		
		// // save the data in the output
		// dualqf2[xcorrIter] = maxval;
		// dual_shiftsInd[xcorrIter] = maxind;
		// dual_freqInd[xcorrIter] = freqlist_inds[maxind];
	// }
	
	// ippsFree(cutout);
	// ippsFree(shifts);
	// ippsFree(dft_in); ippsFree(dft_out);
	// ippsFree(magnSq);
	// ippsFree(conv_out);
	
	// ippsFree(productpeaks);
	// ippsFree(freqlist_inds);
	
	// _endthreadex(0);
    // return 0;
// }


/* The gateway function */
// calling arguments, out = (signal, f_tap, N, Dec)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// initialize with a particular card
	cudaSetDevice(0);
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	printf("Current GPU Device: %s\n", prop.name);
	
	/* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 Inputs required.");
    }
	
	// check inputs from matlab (type safety)
	if (!mxIsInt32(prhs[3])){
		mexErrMsgTxt("ERROR: Shifts must be type int32.");
	}
	
	if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1])){
		mexErrMsgTxt("ERROR: Cutout and y must be type single-precision.");
	}
	
    // declare input variables from matlab
	mxComplexSingle *cutout = mxGetComplexSingles(prhs[0]);
	mxComplexSingle *y = mxGetComplexSingles(prhs[1]);
	float cutout_pwr = (float)mxGetScalar(prhs[2]);
	int *shifts = (int*)mxGetInt32s(prhs[3]);
	int BATCH_SIZE = (int)mxGetScalar(prhs[4]);
	
	// lengths of arrays
	int fftlen = (int)mxGetNumberOfElements(prhs[0]);
	int ylen = (int)mxGetNumberOfElements(prhs[1]);
	int shiftPts = (int)mxGetNumberOfElements(prhs[3]);
	
	// check where to copy by looking at the first and last index slices
	int firstIdx = shifts[0] - 1;
	int lastIdx = shifts[shiftPts-1] - 1; // -1 to convert from matlab indexing to C indexing
	Npp32fc *h_yslice = (Npp32fc*)&y[firstIdx]; // set the pointer of where to start copying y
	int ylen_to_copy = lastIdx - firstIdx + (int)fftlen;
	// additional primitive buffers required
	int normBufferSize;
	nppsNormL2GetBufferSize_32fc64f(fftlen, &normBufferSize);
	int maxIndxBufferSize;
	nppsMaxIndxGetBufferSize_32f(fftlen, &maxIndxBufferSize);
	// check whether total memory will be sufficient
	size_t totalMemReq = (ylen_to_copy + fftlen * BATCH_SIZE + fftlen) * sizeof(Npp32fc)
						+ shiftPts * sizeof(int) * 2 
						+ (shiftPts + BATCH_SIZE*fftlen + BATCH_SIZE)*sizeof(Npp32f) 
						+ normBufferSize * BATCH_SIZE 
						+ maxIndxBufferSize * BATCH_SIZE;
	size_t cufftWorksize;
	int N[1] = {fftlen};
	int istride = 1;
	int inembed[1] = {fftlen * BATCH_SIZE};
	int idist = fftlen;
	int ostride = 1;
	int onembed[1] = {fftlen * BATCH_SIZE};
	int odist = fftlen;
	cufftEstimateMany(1, N, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, BATCH_SIZE, &cufftWorksize);
	
	printf("Total device memory required = %lld/%lld bytes.\n", totalMemReq, MAX_GPU_MEMORY_ALLOC);
	printf("Total extra memory for cufft worksize = %lld bytes. \n", cufftWorksize);
	if (totalMemReq + cufftWorksize > MAX_GPU_MEMORY_ALLOC){
		mexErrMsgTxt("ERROR: Max memory allocation breached. Try reducing batch size.");
	}
	
	// declare equivalent device input variables
	Npp32fc *d_cutout, *d_yslice;
	int *d_shifts;
	Npp32fc *d_fft_inout;
	Npp64f *d_y_norm;
	Npp32f *d_y_norm_32f;
	Npp32f *d_powerspectr;
	// allocate device memory for inputs if memory requirements are fine
	cudaMalloc((void**)&d_cutout, sizeof(Npp32fc)*fftlen);
	cudaMalloc((void**)&d_yslice, sizeof(Npp32fc)*ylen_to_copy);
	cudaMalloc((void**)&d_shifts, sizeof(int)*shiftPts);
	cudaMalloc((void**)&d_fft_inout, sizeof(Npp32fc)*fftlen*BATCH_SIZE); // this is the workspace for the ffts
	cudaMalloc((void**)&d_y_norm, sizeof(Npp64f)*BATCH_SIZE); // workspace for calculating L2 norm squares of the section of y_pwr
	cudaMalloc((void**)&d_y_norm_32f, sizeof(Npp32f)*BATCH_SIZE); // convert to 32f after
	cudaMalloc((void**)&d_powerspectr, sizeof(Npp32f)*fftlen*BATCH_SIZE); // workspace for calculating the abs squares
	// more device memory for buffers required
	Npp8u *d_normBuffer, *d_maxIndxBuffer;
	cudaMalloc((void**)&d_normBuffer, normBufferSize*BATCH_SIZE);
	cudaMalloc((void**)&d_maxIndxBuffer, maxIndxBufferSize*BATCH_SIZE);
	

	// //reserve stuff for threads
    // int t; 
    // HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
	
	/* create the output matrix for matlab */
    plhs[0] = mxCreateNumericMatrix(1,shiftPts, mxSINGLE_CLASS,mxREAL);
	float *h_productpeaks = mxGetSingles(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1,shiftPts, mxINT32_CLASS,mxREAL);
	int *h_freqlist_inds = mxGetInt32s(plhs[1]);
	
	// create equivalent device output
	Npp32f *d_productpeaks;
	int *d_freqlist_inds;
	cudaMalloc((void**)&d_productpeaks, sizeof(Npp32f)*shiftPts);
	cudaMalloc((void**)&d_freqlist_inds, sizeof(int)*shiftPts);
    
	// cufft parameters, batchmode, casts to int from size_t to suppress warnings
	cufftHandle batchplan;
	// int N[1] = {fftlen};
	// int istride = 1;
	// int inembed[1] = {fftlen * BATCH_SIZE};
	// int idist = fftlen;
	// int ostride = 1;
	// int onembed[1] = {fftlen * BATCH_SIZE};
	// int odist = fftlen; // all declared up there
	cufftResult cfr = cufftPlanMany(&batchplan,1,N,
				   inembed,istride,idist,
				   onembed,ostride,odist,
				   CUFFT_C2C,BATCH_SIZE);
	if (cfr != CUFFT_SUCCESS){
		printf("Error: %i\n", cfr);
		mexErrMsgTxt("cufft Plan failed!");
	}
	else{printf("cufft Planning complete.\n");}
	
	// === PRE-COMPUTATION COPIES
	cudaError_t ce_t;
	
	ce_t = cudaMemcpy(d_cutout,cutout,fftlen*sizeof(Npp32fc), cudaMemcpyHostToDevice);
	if (ce_t != cudaSuccess){printf("Failed to copy cutout\n");}
	ce_t = cudaMemcpy(d_yslice,h_yslice,ylen_to_copy*sizeof(Npp32fc), cudaMemcpyHostToDevice);
	if (ce_t != cudaSuccess){printf("Failed to copy yslice\n");}
	// cudaMemcpy(d_shifts,shifts,shiftPts*sizeof(int), cudaMemcpyHostToDevice);
	
	// === COMPUTATIONS ====
	// debugging
	NppStatus ns;
	
	// main loop
	int i,b,CUR_BATCH_SIZE;
	int shift;
	int numBlocks;
	for (i=0; i<shiftPts; i=i+BATCH_SIZE){
		
		CUR_BATCH_SIZE = BATCH_SIZE;
		if (shiftPts - i < BATCH_SIZE){
			CUR_BATCH_SIZE = shiftPts - i;
		}
		
		// fill up the batched workspaces
		for (b=0; b<CUR_BATCH_SIZE; b++){
			// get the index for each iteration
			shift = shifts[i + b] - 1 - firstIdx; // -1 and remove initial offset
			
			// make the multiplies into the fft workspace
			ns = nppsMul_32fc(d_cutout, &d_yslice[shift], &d_fft_inout[b*fftlen], fftlen);
			// if (ns < 0){
				// printf("i = %i, b = %i, NppStatus error : %i\n", i, b, ns);
			// }
			
			// get the norms of the y sections
			nppsNorm_L2_32fc64f(&d_yslice[shift], fftlen, &d_y_norm[b], &d_normBuffer[normBufferSize*b]);
		} //end of loop over batch size
		
		// run the ffts
		cufftExecC2C(batchplan, (cufftComplex*)d_fft_inout, (cufftComplex*)d_fft_inout, CUFFT_FORWARD); // in-place
		
		// run the kernel to get the power spectrum of the ffts (on the whole batch)
		numBlocks = (fftlen * CUR_BATCH_SIZE) / THREADS_PER_BLOCK + 1;
		power_spectr_kernel<<< numBlocks, THREADS_PER_BLOCK>>>(fftlen * CUR_BATCH_SIZE, d_fft_inout, d_powerspectr);
		
		for (b=0; b<CUR_BATCH_SIZE;b++){
			// find the max and maxindex of the power spectrum
			nppsMaxIndx_32f(&d_powerspectr[b*fftlen], fftlen, &d_productpeaks[i + b], &d_freqlist_inds[i + b], &d_maxIndxBuffer[maxIndxBufferSize*b]);
			
			
		} // end of loop over batch size
		
		// scale this batch's output by the norms		
		nppsDivC_32f_I((Npp32f)cutout_pwr, &d_productpeaks[i], CUR_BATCH_SIZE); // divide by cutout pwr
		nppsConvert_64f32f(d_y_norm, d_y_norm_32f, CUR_BATCH_SIZE); // convert norm values to 32f
		nppsDiv_32f_I(d_y_norm_32f, &d_productpeaks[i], CUR_BATCH_SIZE); 
		nppsDiv_32f_I(d_y_norm_32f, &d_productpeaks[i], CUR_BATCH_SIZE); // divide twice because we didnt square it (this is so bad but whatever works for now)
		
	} // end of main loop
	
	// copy output back
	ce_t = cudaMemcpy(h_productpeaks, d_productpeaks, shiftPts*sizeof(Npp32f), cudaMemcpyDeviceToHost);
	if (ce_t != cudaSuccess){
		printf("Failed to copy productpeaks\n");
		if (ce_t == cudaErrorInvalidDevicePointer){
			printf("Invalid device pointer\n");
		}
		else if(ce_t == cudaErrorInvalidValue){printf("Invalid value\n");}
		else if(ce_t == cudaErrorInvalidMemcpyDirection){printf("Invalid copy direction\n");}
	}
	ce_t = cudaMemcpy(h_freqlist_inds, d_freqlist_inds, shiftPts*sizeof(int), cudaMemcpyDeviceToHost);
	if (ce_t != cudaSuccess){printf("Failed to copy freqlistinds\n");}
	
	// === END OF COMPUTATIONS ===
	
	// cleanup
	cudaFree(d_cutout);
	cudaFree(d_yslice);
	cudaFree(d_shifts);
	cudaFree(d_fft_inout);
	cudaFree(d_y_norm);
	cudaFree(d_y_norm_32f);
	cudaFree(d_powerspectr);
	
	cudaFree(d_normBuffer);
	cudaFree(d_maxIndxBuffer);
	
	cudaFree(d_productpeaks);
	cudaFree(d_freqlist_inds);
	
	cufftDestroy(batchplan);
	
}