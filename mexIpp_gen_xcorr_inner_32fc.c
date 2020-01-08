#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"
#include <windows.h>
#include <process.h>

#define NUM_THREADS 24

void ifftshift_32fc(Ipp32fc *in, Ipp32fc *out, int len){
	if (len%2 == 0){ // even
		memcpy(&out[len/2], &in[0], (len/2) * sizeof(Ipp32fc));
		memcpy(&out[0], &in[len/2], (len/2) * sizeof(Ipp32fc));
	}
	else{ // out
		memcpy(&out[len/2], &in[0], (len/2) * sizeof(Ipp32fc)); // this should round down due to integer divisions i.e. len/2 is less than half
		memcpy(&out[0], &in[len/2], (len/2 + 1) * sizeof(Ipp32fc)); // note we need the +1 here
	}
}

void ippsSelfMedian_32f_I(Ipp32f *pSrcDst, int len, Ipp32f *median){
	ippsSortAscend_32f_I(pSrcDst, len);
	if (len%2 == 0){ // if it's even then take the two middle and centre of those two values
		*median = (pSrcDst[len/2 - 1] + pSrcDst[len/2]) / 2;
	}
	else{ // if it's odd then just take the middle one
		*median = pSrcDst[len/2]; // signed integer division will truncate towards zero
	}
}
	
struct thread_data{
	int thread_t_ID;
	
	mxComplexSingle *thread_rx_fft;
	mxComplexSingle *thread_cutout_fft_windowed_conjds;
	int *thread_dsIdx_matlab;
	int *thread_windowIdx_matlab;
	int thread_freqDS_factor;
	int thread_numBins;
	int thread_cutoutlen_ds;
	int thread_numFreqIters;
	float thread_cutoutNorm_dsSq;
	int thread_rx_fft_len;
	int thread_dsIdx_len;
	
	float *thread_qf2_surface;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_ipps_gen_xcorr_inner(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	mxComplexSingle *rx_fft = inner_data->thread_rx_fft;
	mxComplexSingle *cutout_fft_windowed_conjds = inner_data->thread_cutout_fft_windowed_conjds;
	int *dsIdx_matlab = inner_data->thread_dsIdx_matlab;
	int *windowIdx_matlab = inner_data->thread_windowIdx_matlab;
	int freqDS_factor = inner_data->thread_freqDS_factor;
	int numBins = inner_data->thread_numBins;
	int cutoutlen_ds = inner_data->thread_cutoutlen_ds;
	int numFreqIters = inner_data->thread_numFreqIters;
	float cutoutNorm_dsSq = inner_data->thread_cutoutNorm_dsSq;
	int rx_fft_len = inner_data->thread_rx_fft_len;
	int dsIdx_len = inner_data->thread_dsIdx_len;
	
	float *qf2_surface = inner_data->thread_qf2_surface;
	// ======= end of assignments

	int windowIdx_start = windowIdx_matlab[0] - 1;
	int clipLen = dsIdx_len - cutoutlen_ds + 1;
	Ipp32f maxq;
	int maxqi;
	Ipp32f medianq;
	int i, k;
	Ipp64f tmpnorm;
	
	Ipp32fc *rx_fft_windowed = ippsMalloc_32fc_L(rx_fft_len);
	Ipp32fc *rx_fft_windowed_ds = &rx_fft_windowed[dsIdx_matlab[0]-1]; // helper pointer
	Ipp32fc *preproduct = ippsMalloc_32fc_L(dsIdx_len);
	Ipp32fc *result = ippsMalloc_32fc_L(dsIdx_len);
	Ipp32fc *rx_windowed_ds = ippsMalloc_32fc_L(dsIdx_len);
	Ipp32f *rxNorm_ds = ippsMalloc_32f_L(clipLen);
	Ipp32f *qf2 = ippsMalloc_32f_L(clipLen);
	
	
	// IPP DFT creation
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;   
	ippsDFTGetSize_C_32fc(dsIdx_len, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf); // this just fills the 3 integers
	/* memory allocation */
	IppsDFTSpec_C_32fc *pSpec = (IppsDFTSpec_C_32fc*)ippMalloc(sizeSpec); // this is analogue of the fftw plan
	Ipp8u *pBuffer = (Ipp8u*)ippMalloc(sizeBuf);
	Ipp8u *pMemInit = (Ipp8u*)ippMalloc(sizeInit);
	ippsDFTInit_C_32fc(dsIdx_len, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone,  pSpec, pMemInit); // inv by N follows the matlab convention

	
	for (i = t_ID; i < numFreqIters; i = i+NUM_THREADS){
		// first we zero the rx_fft_windowed
		ippsZero_32fc(rx_fft_windowed, dsIdx_len);
		// copy the windowed part of rx_fft into this
		ippsCopy_32fc((Ipp32fc*)&rx_fft[i * freqDS_factor], (Ipp32fc*)&rx_fft_windowed[windowIdx_start], numBins);
		// downsample it by pointing it to a certain index (already done since its always the same)
		
		// Calculate preproduct
		ippsMul_32fc((Ipp32fc*)rx_fft_windowed_ds, (Ipp32fc*)cutout_fft_windowed_conjds, preproduct, dsIdx_len);
		
		ippsDFTInv_CToC_32fc(preproduct, result, pSpec, pBuffer);
		
		// Calculate normalization
		ippsDFTInv_CToC_32fc(rx_fft_windowed_ds, rx_windowed_ds, pSpec, pBuffer);
		
		for (k = 0; k < clipLen; k++){
			ippsNorm_L2_32fc64f(&rx_windowed_ds[k], cutoutlen_ds, &tmpnorm);
			rxNorm_ds[k] = (Ipp32f)tmpnorm; // cast it after
			
		}
		
		// Convert normalization to squared
		ippsSqr_32f_I(rxNorm_ds, clipLen);
		
		// Normalize and output qf2
		ippsPowerSpectr_32fc(result, qf2, clipLen);
		ippsDivC_32f_I(cutoutNorm_dsSq, qf2, clipLen);
		ippsDiv_32f_I(rxNorm_ds, qf2, clipLen);
		
		// Now save the data into the surface
		ippsMaxIndx_32f(qf2, clipLen, &maxq, &maxqi);
		ippsSelfMedian_32f_I(qf2, clipLen, &medianq); // note that after this line the qf2 will have been resorted
		
		// Arrange into the rows
		qf2_surface[3*i + 0] = (float)maxq;
		qf2_surface[3*i + 1] = (float)(maxq/(1-maxq)/medianq);
		
	}
		
	//freeing
	ippsFree(rx_fft_windowed);
	ippsFree(preproduct);
	ippsFree(result);
	ippsFree(rx_windowed_ds);
	ippsFree(rxNorm_ds);
	ippsFree(qf2);
	
	ippFree(pBuffer);
	ippFree(pMemInit);
	ippFree(pSpec);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare inputs
	mxComplexSingle *rx_fft;
	int rx_fft_len;
	mxComplexSingle *cutout_fft_windowed_conjds; // same length as dsIdx
	int *dsIdx_matlab;
	int dsIdx_len;
	int *windowIdx_matlab; // length is numBins
	int freqDS_factor;
	int numBins;
	int cutoutlen_ds;
	int numFreqIters;
	float cutoutNorm_dsSq;
	
	// declare outputs
	float *qf2_surface;

	int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads

	
    /* check for proper number of arguments */
    if (nrhs!=9){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","9 Inputs required.");
    }

	rx_fft = mxGetComplexSingles(prhs[0]);
	cutout_fft_windowed_conjds = mxGetComplexSingles(prhs[1]);
	dsIdx_matlab = mxGetInt32s(prhs[2]);
	windowIdx_matlab = mxGetInt32s(prhs[3]);
	freqDS_factor = (int)mxGetScalar(prhs[4]);
	numBins = (int)mxGetScalar(prhs[5]);
	cutoutlen_ds = (int)mxGetScalar(prhs[6]);
	numFreqIters = (int)mxGetScalar(prhs[7]);
	cutoutNorm_dsSq = (float)mxGetScalar(prhs[8]);
	
	rx_fft_len = (int)mxGetM(prhs[0]) * (int)mxGetN(prhs[0]);
	dsIdx_len = (int)mxGetM(prhs[1]) * (int)mxGetN(prhs[1]);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateNumericMatrix(3,numFreqIters, mxSINGLE_CLASS,mxREAL);
    /* get a pointer to the real data in the output matrix */
    qf2_surface = mxGetSingles(plhs[0]);
	
	// =============/* call the computational routine */==============
	for (t = 0; t<NUM_THREADS; t++){
		// attach input arguments
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_rx_fft = rx_fft;
		thread_data_array[t].thread_cutout_fft_windowed_conjds = cutout_fft_windowed_conjds;
		thread_data_array[t].thread_dsIdx_matlab = dsIdx_matlab;
		thread_data_array[t].thread_windowIdx_matlab = windowIdx_matlab;
		thread_data_array[t].thread_freqDS_factor = freqDS_factor;
		thread_data_array[t].thread_numBins = numBins;
		thread_data_array[t].thread_cutoutlen_ds = cutoutlen_ds;
		thread_data_array[t].thread_numFreqIters = numFreqIters;
		thread_data_array[t].thread_cutoutNorm_dsSq = cutoutNorm_dsSq;
		thread_data_array[t].thread_rx_fft_len = rx_fft_len;
		thread_data_array[t].thread_dsIdx_len = dsIdx_len;
		
		thread_data_array[t].thread_qf2_surface = qf2_surface;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ipps_gen_xcorr_inner,(void*)&thread_data_array[t],0,NULL);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);

	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
    }
    // printf("All threads closed! \n");
	// =====================================
	
}