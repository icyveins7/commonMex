#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"
// #include <immintrin.h> // include if trying the intrinsics code

#define NUM_THREADS 40

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

struct t_data{
	int thread_t_ID;
	
	mxComplexSingle *thread_x;
	mxComplexSingle *thread_y;
	int *thread_startIdx_list;
	int thread_xcorrIterTotal;
	int *thread_shiftsZero;
	int thread_shiftPts;
	int thread_mullen;
	int thread_nfft;
	
	mxComplexSingle *thread_ww;
	mxComplexSingle *thread_aa;
	mxComplexSingle *thread_fv;
	int thread_k_bins;
	
	// IPP DFT vars
	Ipp8u *thread_pBuffer;
	IppsDFTSpec_C_32fc *thread_pSpec;
	
	float *thread_dualqf2;
	int *thread_dual_shiftsInd;
	int *thread_dual_freqInd;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

unsigned __stdcall xcorr_dualChannel(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	mxComplexSingle *x = inner_data->thread_x;
	mxComplexSingle *y = inner_data->thread_y;
	int *startIdx_list = inner_data->thread_startIdx_list; // list of startIdx to go to
	int xcorrIterTotal = inner_data->thread_xcorrIterTotal;
	int *shiftsZero = inner_data->thread_shiftsZero;
	int shiftPts = inner_data->thread_shiftPts;
	int mullen = inner_data->thread_mullen; // technically no longer the actual fftlen, but rather the multiplylen
	int nfft = inner_data->thread_nfft;
	
	mxComplexSingle *ww = inner_data->thread_ww;
	mxComplexSingle *aa = inner_data->thread_aa;
	mxComplexSingle *fv = inner_data->thread_fv;
	int k_bins = inner_data->thread_k_bins;
	
	// IPP DFT vars
	Ipp8u *pBuffer = inner_data->thread_pBuffer;
	IppsDFTSpec_C_32fc *pSpec = inner_data->thread_pSpec;
	
	float *dualqf2 = inner_data->thread_dualqf2;
	int *dual_shiftsInd = inner_data->thread_dual_shiftsInd;
	int *dual_freqInd = inner_data->thread_dual_freqInd;
	// end of attached variables

	// computations
	int i, xcorrIter, curr_shift;
	int startIdx;
	float cutout_pwr;
	double cutout_pwr_64f;
	float y_pwr;
	double y_pwr_64f;
	Ipp32s *shifts = (Ipp32s*)ippsMalloc_32s_L(shiftPts);
	Ipp32fc *cutout = (Ipp32fc*)ippsMalloc_32fc_L(mullen);
	Ipp32fc *dft_in = (Ipp32fc*)ippsMalloc_32fc_L(nfft);
	Ipp32fc *dft_out = (Ipp32fc*)ippsMalloc_32fc_L(nfft);
	Ipp32fc *conv_out = (Ipp32fc*)ippsMalloc_32fc_L(nfft);
	Ipp32f *magnSq = (Ipp32f*)ippsMalloc_32f_L(k_bins);
	Ipp32f maxval;
	int maxind;
	
	// temp arrays for single xcorr equivalent
	Ipp32f *productpeaks = (Ipp32f*)ippsMalloc_32f_L(shiftPts);
	Ipp32s *freqlist_inds = (Ipp32s*)ippsMalloc_32s_L(shiftPts);
	
	// zero out the dft_in so the padded part is all zeroes
	ippsZero_32fc(dft_in, nfft);
	
	// pick xcorrIter based on thread number
	for (xcorrIter = t_ID; xcorrIter<xcorrIterTotal; xcorrIter = xcorrIter + NUM_THREADS){
		startIdx = startIdx_list[xcorrIter];
		ippsAddC_32s_Sfs((Ipp32s*)shiftsZero, (Ipp32s)startIdx, shifts, shiftPts, 0); // scale factor of 0 implies the same value
		ippsConj_32fc((Ipp32fc*)&x[startIdx-1],cutout, mullen); // need to save the conjugate in cutout, need -1 to convert from matlab indexing
		ippsNorm_L2_32fc64f((Ipp32fc*)&x[startIdx-1], mullen, &cutout_pwr_64f);
		cutout_pwr = (float)(cutout_pwr_64f*cutout_pwr_64f);
		
		for (i = 0; i<shiftPts; i++){
			curr_shift = shifts[i]-1;
			// printf("Working on thread %i, loop %i, shift %i \n",thread,i,curr_shift);
			
			ippsNorm_L2_32fc64f((Ipp32fc*)&y[curr_shift], mullen, &y_pwr_64f);
			y_pwr = (float)(y_pwr_64f*y_pwr_64f);
			
			ippsMul_32fc((Ipp32fc*)cutout,(Ipp32fc*)&y[curr_shift], (Ipp32fc*)dft_in, mullen); // so we multiply the short mullen
			
			// now do the convolution!
			ippsMul_32fc_I((Ipp32fc*)aa, dft_in, mullen);
			
			ippsDFTFwd_CToC_32fc(dft_in, dft_out, pSpec, pBuffer); // but we dft the longer nfft
			
			ippsMul_32fc_I((Ipp32fc*)fv, dft_out, nfft);
			
			ippsDFTInv_CToC_32fc(dft_out, conv_out, pSpec, pBuffer);
			
			ippsMul_32fc(&conv_out[mullen-1], (Ipp32fc*)&ww[mullen-1], dft_out, k_bins); // we reuse the start of dft_out to store the shorter k_bins
			
			ippsPowerSpectr_32fc(dft_out, magnSq, k_bins); // we calculate magnSq of the start of dft_out, only k_bins length
			
			ippsMaxIndx_32f(magnSq, k_bins, &maxval, &maxind);
			
			productpeaks[i] = maxval/cutout_pwr/y_pwr;
			freqlist_inds[i] = maxind;
		}
		
		// now find the max in the single productpeaks
		ippsMaxIndx_32f(productpeaks, shiftPts, &maxval, &maxind);
		
		// save the data in the output
		dualqf2[xcorrIter] = maxval;
		dual_shiftsInd[xcorrIter] = maxind;
		dual_freqInd[xcorrIter] = freqlist_inds[maxind];
	}
	
	ippsFree(cutout);
	ippsFree(shifts);
	ippsFree(dft_in); ippsFree(dft_out);
	ippsFree(magnSq);
	ippsFree(conv_out);
	
	ippsFree(productpeaks);
	ippsFree(freqlist_inds);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	// == INITIALIZE TIMING ==
	int start_t = StartCounter();
	int end_t;
	
	ippInit();
	
	// struct timespec vartime;
	double totalTime;
	
    // declare variables
    mxComplexSingle *x, *y;
	int *startIdx_list;
	int xcorrIterTotal;
    int *shiftsZero; // direct matlab inputs are always in floats
	int	shiftPts;
	int mullen, nfft;
	mxComplexSingle *ww, *aa, *fv;
	int k_bins;
	// declare outputs
	float *dualqf2;
	int *dual_shiftsInd;
	int *dual_freqInd;

	// //reserve stuff for threads
    int t; 
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=10){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","10 Inputs required.");
    }

    x = mxGetComplexSingles(prhs[0]); 
	y = mxGetComplexSingles(prhs[1]); 
	startIdx_list = mxGetInt32s(prhs[2]);
	shiftsZero = mxGetInt32s(prhs[3]);
	mullen = (int)mxGetScalar(prhs[4]);// this is the length of the fft
    nfft = (int)mxGetScalar(prhs[5]);// this is the length of the fft
	ww = mxGetComplexSingles(prhs[6]);
	aa = mxGetComplexSingles(prhs[7]);
	fv = mxGetComplexSingles(prhs[8]);
	k_bins = (int)mxGetScalar(prhs[9]);
	
	xcorrIterTotal = (int)mxGetM(prhs[2]) * (int)mxGetN(prhs[2]);
	shiftPts = (int)mxGetN(prhs[3]); // this is the number of shifts there are
	
	/* create the output matrix */
    plhs[0] = mxCreateNumericMatrix(1,xcorrIterTotal, mxSINGLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(1,xcorrIterTotal, mxINT32_CLASS,mxREAL); //initializes to 0
	plhs[2] = mxCreateNumericMatrix(1,xcorrIterTotal, mxINT32_CLASS,mxREAL); //initializes to 0
    
    /* get a pointer to the real data in the output matrix */
    dualqf2 = mxGetSingles(plhs[0]);
    dual_shiftsInd = (int*)mxGetInt32s(plhs[1]);
	dual_freqInd = (int*)mxGetInt32s(plhs[2]);
	
	end_t = GetCounter();
	totalTime = (end_t - start_t)/PCFreq; // in ms
	printf("Time for initial handling = %g ms \n",totalTime);
	
	// ===== IPP DFT Allocations =====
	start_t = GetCounter();
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;   
	ippsDFTGetSize_C_32fc(nfft, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf); // this just fills the 3 integers
	/* memory allocation */
	IppsDFTSpec_C_32fc **pSpec = (IppsDFTSpec_C_32fc**)ippMalloc(sizeof(IppsDFTSpec_C_32fc*)*NUM_THREADS);
	Ipp8u **pBuffer = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	Ipp8u **pMemInit = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	for (t = 0; t<NUM_THREADS; t++){ // make one for each thread
		pSpec[t] = (IppsDFTSpec_C_32fc*)ippMalloc(sizeSpec); // this is analogue of the fftw plan
		pBuffer[t] = (Ipp8u*)ippMalloc(sizeBuf);
		pMemInit[t] = (Ipp8u*)ippMalloc(sizeInit);
		ippsDFTInit_C_32fc(nfft, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone,  pSpec[t], pMemInit[t]); // kinda like making the fftw plan?
	}
	end_t = GetCounter();
	totalTime = (end_t - start_t)/PCFreq; // in ms
	printf("Time to prepare IPP DFTs = %g ms \n",totalTime);
	// ================================================================
	start_t = GetCounter();

    // =============/* call the computational routine */==============
	GROUP_AFFINITY currentGroupAffinity, newGroupAffinity;
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_x = x;
		t_data_array[t].thread_y = y;
		t_data_array[t].thread_startIdx_list = startIdx_list;
		t_data_array[t].thread_xcorrIterTotal = xcorrIterTotal;
		t_data_array[t].thread_shiftsZero = shiftsZero;
		t_data_array[t].thread_shiftPts = shiftPts;
		t_data_array[t].thread_mullen = mullen;
		t_data_array[t].thread_nfft = nfft;
		
		t_data_array[t].thread_ww = ww;
		t_data_array[t].thread_aa = aa;
		t_data_array[t].thread_fv = fv;
		t_data_array[t].thread_k_bins = k_bins;
			
			
		// IPP DFT vars
		t_data_array[t].thread_pBuffer = pBuffer[t];
		t_data_array[t].thread_pSpec = pSpec[t];
		
		t_data_array[t].thread_dualqf2 = dualqf2;
		t_data_array[t].thread_dual_shiftsInd = dual_shiftsInd;
		t_data_array[t].thread_dual_freqInd = dual_freqInd;
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&xcorr_dualChannel,(void*)&t_data_array[t],0,NULL);
		// GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
		// newGroupAffinity = currentGroupAffinity;
		// newGroupAffinity.Group = t%2;
		// SetThreadGroupAffinity(ThreadList[t], &newGroupAffinity, NULL);
		// GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
        // printf("Beginning threadID %i, group %i..\n",t_data_array[t].thread_t_ID, currentGroupAffinity.Group);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
	
	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    printf("All threads closed! \n");
	end_t = GetCounter();
	totalTime = (end_t - start_t)/PCFreq; // in ms
	printf("Time for threads to finish = %g ms \n",totalTime);
	
	for (t=0; t<NUM_THREADS; t++){
		ippFree(pSpec[t]);
		ippFree(pBuffer[t]);
		ippFree(pMemInit[t]);
	}
	ippFree(pSpec);
	ippFree(pBuffer);
	ippFree(pMemInit);
	
	end_t = GetCounter();
	totalTime = (end_t - start_t)/PCFreq; // in ms
	printf("Time for threads to finish and free stuff = %g ms \n",totalTime);
}