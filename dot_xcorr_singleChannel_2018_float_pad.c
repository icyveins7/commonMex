#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"
// #include <immintrin.h> // include if trying the intrinsics code

#define NUM_THREADS 12 // seems like a good number now, you may increase if your computation time takes more than 1 second, otherwise spawning more threads takes longer than the actual execution time per thread lol

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
	
	mxComplexSingle *thread_cutout;
	mxComplexSingle *thread_y;
	float thread_cutout_pwr;
	float *thread_shifts;
	int thread_shiftPts;
	int thread_mullen;
	int thread_padlen;
	
	// IPP DFT vars
	Ipp8u *thread_pBuffer;
	IppsDFTSpec_C_32fc *thread_pSpec;
	
	float *thread_productpeaks;
	int *thread_freqlist_inds;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

unsigned __stdcall xcorr_singleChannel(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	mxComplexSingle *cutout = inner_data->thread_cutout;
	mxComplexSingle *y = inner_data->thread_y;
	float cutout_pwr = inner_data->thread_cutout_pwr;
	float *shifts = inner_data->thread_shifts;
	int shiftPts = inner_data->thread_shiftPts;
	int mullen = inner_data->thread_mullen; // technically no longer the actual fftlen, but rather the multiplylen
	int padlen = inner_data->thread_padlen;
	
	// IPP DFT vars
	Ipp8u *pBuffer = inner_data->thread_pBuffer;
	IppsDFTSpec_C_32fc *pSpec = inner_data->thread_pSpec;
	
	float *productpeaks = inner_data->thread_productpeaks;
	int *freqlist_inds = inner_data->thread_freqlist_inds;
	// end of attached variables

	// computations
	int i, curr_shift;
	float y_pwr;
	double y_pwr_64f;
	Ipp32fc *dft_in = (Ipp32fc*)ippsMalloc_32fc_L(padlen);
	Ipp32fc *dft_out = (Ipp32fc*)ippsMalloc_32fc_L(padlen);
	Ipp32f *magnSq = (Ipp32f*)ippsMalloc_32f_L(padlen);
	Ipp32f maxval;
	int maxind;
	
	// zero out the dft_in so the padded part is all zeroes
	ippsZero_32fc(dft_in, padlen);
	
	// pick point based on thread number
	for (i = t_ID; i<shiftPts; i=i+NUM_THREADS){
		curr_shift = (int)shifts[i]-1;
		// printf("Working on thread %i, loop %i, shift %i \n",thread,i,curr_shift);
		
		ippsNorm_L2_32fc64f((Ipp32fc*)&y[curr_shift], mullen, &y_pwr_64f);
		y_pwr = (float)(y_pwr_64f*y_pwr_64f);
		
        ippsMul_32fc((Ipp32fc*)cutout,(Ipp32fc*)&y[curr_shift], (Ipp32fc*)dft_in, mullen); // so we multiply the short mullen
		
		ippsDFTFwd_CToC_32fc(dft_in, dft_out, pSpec, pBuffer); // but we dft the longer padlen
		
		ippsPowerSpectr_32fc(dft_out, magnSq, padlen); // and all consequent calculations are done with the longer padlen
		
		ippsMaxIndx_32f(magnSq, padlen, &maxval, &maxind);
		
		productpeaks[i] = maxval/cutout_pwr/y_pwr;
		freqlist_inds[i] = maxind;
	}
	
	ippsFree(dft_in); ippsFree(dft_out);
	ippsFree(magnSq);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
	// == INITIALIZE TIMING ==
	int start_t = StartCounter();
	int end_t;
	// struct timespec vartime;
	double totalTime;
	
    // declare variables
    mxComplexSingle *cutout, *y;
    float *shifts; // direct matlab inputs are always in floats
	float cutout_pwr;
	int	shiftPts;
	int m, mullen, padlen;
	// declare outputs
	float *productpeaks;
	int *freqlist_inds; // declared to be int below

	// //reserve stuff for threads
    int t; 
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 Inputs required.");
    }

    cutout = mxGetComplexSingles(prhs[0]); 
	y = mxGetComplexSingles(prhs[1]); 
	cutout_pwr = (float)mxGetScalar(prhs[2]);
	shifts = mxGetSingles(prhs[3]);
    padlen = (int)mxGetScalar(prhs[4]);// this is the length of the fft
	
    m = (int)mxGetM(prhs[0]);
    mullen = m*(int)mxGetN(prhs[0]); // length of the cutout, for multiplies
	
	
	shiftPts = (int)mxGetN(prhs[3]); // this is the number of shifts there are
	
	/* create the output matrix */
    plhs[0] = mxCreateNumericMatrix(1,(int)shiftPts, mxSINGLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, (int)shiftPts, mxINT32_CLASS,mxREAL); //initializes to 0
    
    /* get a pointer to the real data in the output matrix */
    productpeaks = mxGetSingles(plhs[0]);
    freqlist_inds = (int*)mxGetInt32s(plhs[1]);
	
	// ===== IPP DFT Allocations =====
	start_t = GetCounter();
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;   
	ippsDFTGetSize_C_32fc(padlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf); // this just fills the 3 integers
	/* memory allocation */
	IppsDFTSpec_C_32fc **pSpec = (IppsDFTSpec_C_32fc**)ippMalloc(sizeof(IppsDFTSpec_C_32fc*)*NUM_THREADS);
	Ipp8u **pBuffer = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	Ipp8u **pMemInit = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	for (t = 0; t<NUM_THREADS; t++){ // make one for each thread
		pSpec[t] = (IppsDFTSpec_C_32fc*)ippMalloc(sizeSpec); // this is analogue of the fftw plan
		pBuffer[t] = (Ipp8u*)ippMalloc(sizeBuf);
		pMemInit[t] = (Ipp8u*)ippMalloc(sizeInit);
		ippsDFTInit_C_32fc(padlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,  pSpec[t], pMemInit[t]); // kinda like making the fftw plan?
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
		
		t_data_array[t].thread_cutout = cutout;
		t_data_array[t].thread_y = y;
		t_data_array[t].thread_cutout_pwr = cutout_pwr;
		t_data_array[t].thread_shifts = shifts;
		t_data_array[t].thread_shiftPts = shiftPts;
		t_data_array[t].thread_mullen = mullen;
		t_data_array[t].thread_padlen = padlen;
			
		// IPP DFT vars
		t_data_array[t].thread_pBuffer = pBuffer[t];
		t_data_array[t].thread_pSpec = pSpec[t];
		
		t_data_array[t].thread_productpeaks = productpeaks;
		t_data_array[t].thread_freqlist_inds = freqlist_inds;
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&xcorr_singleChannel,(void*)&t_data_array[t],0,NULL);
		// GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
		// newGroupAffinity = currentGroupAffinity;
		// newGroupAffinity.Group = t%2;
		// SetThreadGroupAffinity(ThreadList[t], &newGroupAffinity, NULL);
		// GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
        // printf("Beginning threadID %i..\n",t_data_array[t].thread_t_ID);
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
}