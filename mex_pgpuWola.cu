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
	
    // declare variables
    Npp16sc *h_indata;
	Npp16sc *d_indata;
	
	float *h_ftap;
	Npp32f *d_ftap;
	
	int datalen; // length of indata
	int L; // length of filter
	int N; // number of channels
	int Dec; // decimation factor
	
	// declare outputs
	int nprimePts;
	Npp32fc *d_outdata;
	mxComplexSingle *h_outdata;
	

	// //reserve stuff for threads
    // int t; 
    // HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
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
	
	
	/* create the output matrix */
	if (datalen%Dec!=0){
		mexErrMsgTxt("ERROR: Input data must be a multiple of supplied decimation factor!");
	}
	nprimePts = datalen/Dec;
    plhs[0] = mxCreateNumericMatrix(N,nprimePts, mxSINGLE_CLASS,mxCOMPLEX);
	h_outdata = mxGetComplexSingles(plhs[0]);
    
    // allocate device memory
	cudaMalloc((void**)&d_indata, sizeof(Npp16sc)*datalen);
	cudaMalloc((void**)&d_ftap, sizeof(Npp32f)*L);
	cudaMalloc((void**)&d_outdata, sizeof(Npp32fc)*nprimePts*N);
	
	// cufft parameters, batchmode
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
	
	// cleanup
	cudaFree(d_indata);
	cudaFree(d_ftap);
	cudaFree(d_outdata);
	
	cufftDestroy(batchplan);
	
}