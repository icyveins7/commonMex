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
	float *thread_power_cumu;
	float thread_cutout_pwr;
	float *thread_shifts;
	int thread_shiftPts;
	int thread_fftlen;
	
	// IPP DFT vars
	Ipp8u *thread_pBuffer;
	IppsDFTSpec_C_32fc *thread_pSpec;
	
	float *thread_out;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

unsigned __stdcall calc_TDFD_ambiguity_func(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	mxComplexSingle *cutout = inner_data->thread_cutout;
	mxComplexSingle *y = inner_data->thread_y;
	float *power_cumu = inner_data->thread_power_cumu;
	float cutout_pwr = inner_data->thread_cutout_pwr;
	float *shifts = inner_data->thread_shifts;
	int shiftPts = inner_data->thread_shiftPts;
	int fftlen = inner_data->thread_fftlen;
	
	// IPP DFT vars
	Ipp8u *pBuffer = inner_data->thread_pBuffer;
	IppsDFTSpec_C_32fc *pSpec = inner_data->thread_pSpec;
	
	float *out = inner_data->thread_out;
	// end of attached variables

	// computations
	int i, curr_shift;
	float y_pwr;
	Ipp32fc *dft_in = (Ipp32fc*)ippsMalloc_32fc_L(fftlen);
	Ipp32fc *dft_out = (Ipp32fc*)ippsMalloc_32fc_L(fftlen);
	Ipp32f *magnSq = (Ipp32f*)ippsMalloc_32f_L(fftlen);
	Ipp32f divConstant;
	
	// pick point based on thread number
	for (i = t_ID; i<shiftPts; i=i+NUM_THREADS){
		curr_shift = (int)shifts[i]-1;
		// printf("Working on thread %i, loop %i, shift %i \n",thread,i,curr_shift);
		if (curr_shift == 0){ y_pwr = power_cumu[fftlen-1];} // unlikely to ever happen, but whatever
		else{ y_pwr = power_cumu[curr_shift + fftlen - 1] - power_cumu[curr_shift - 1];}
		
        ippsMul_32fc((Ipp32fc*)cutout,(Ipp32fc*)&y[curr_shift], (Ipp32fc*)dft_in, fftlen);
		
		ippsDFTFwd_CToC_32fc(dft_in, dft_out, pSpec, pBuffer);
		
		ippsPowerSpectr_32fc(dft_out, magnSq, fftlen);
		
		divConstant = cutout_pwr*y_pwr;
		
		ippsDivC_32f(magnSq, divConstant, &out[i*fftlen], fftlen); // write output back to 'out' directly
		
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
    float *power_cumu, *shifts; // direct matlab inputs are always in Singles
	float cutout_pwr;
	int	shiftPts;
	int m, fftlen;
	// declare outputs
	float *out;

	// //reserve stuff for threads
    int t; 
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
    }

    cutout = mxGetComplexSingles(prhs[0]); 
	y = mxGetComplexSingles(prhs[1]); 
	power_cumu = mxGetSingles(prhs[2]);
	cutout_pwr = (float)mxGetScalar(prhs[3]);
	shifts = mxGetSingles(prhs[4]);
    
    m = (int)mxGetM(prhs[0]);
    fftlen = m*(int)mxGetN(prhs[0]); // this is the length of the fft, assume its only 1-D so we just take the product
	
	shiftPts = (int)mxGetN(prhs[4]); // this is the number of shifts there are
	
	/* create the output matrix */
    plhs[0] = mxCreateUninitNumericMatrix(fftlen,shiftPts,mxSINGLE_CLASS,mxREAL);
    
    /* get a pointer to the real data in the output matrix */
    out = mxGetSingles(plhs[0]);
	
	// ===== IPP DFT Allocations =====
	start_t = GetCounter();
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;   
	ippsDFTGetSize_C_32fc(fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf); // this just fills the 3 integers
	/* memory allocation */
	IppsDFTSpec_C_32fc **pSpec = (IppsDFTSpec_C_32fc**)ippMalloc(sizeof(IppsDFTSpec_C_32fc*)*NUM_THREADS);
	Ipp8u **pBuffer = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	Ipp8u **pMemInit = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	for (t = 0; t<NUM_THREADS; t++){ // make one for each thread
		pSpec[t] = (IppsDFTSpec_C_32fc*)ippMalloc(sizeSpec); // this is analogue of the fftw plan
		pBuffer[t] = (Ipp8u*)ippMalloc(sizeBuf);
		pMemInit[t] = (Ipp8u*)ippMalloc(sizeInit);
		ippsDFTInit_C_32fc(fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,  pSpec[t], pMemInit[t]); // kinda like making the fftw plan?
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
		t_data_array[t].thread_power_cumu = power_cumu;
		t_data_array[t].thread_cutout_pwr = cutout_pwr;
		t_data_array[t].thread_shifts = shifts;
		t_data_array[t].thread_shiftPts = shiftPts;
		t_data_array[t].thread_fftlen = fftlen;
			
		// IPP DFT vars
		t_data_array[t].thread_pBuffer = pBuffer[t];
		t_data_array[t].thread_pSpec = pSpec[t];
		
		t_data_array[t].thread_out = out;
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&calc_TDFD_ambiguity_func,(void*)&t_data_array[t],0,NULL);
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
