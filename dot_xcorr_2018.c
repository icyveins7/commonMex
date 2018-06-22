// this is a modification of testcomplex; small bandwidth with high number of channels results in poor use of processor power
// it is better to assign each thread to then solve each channel instead?

// Call via:
// [allproductpeaks, allfreqlist_inds] = dot_xcorr(conjcutout, channels, numChans, cutout_pwr, shifts);

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"

#define NUM_THREADS 24

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
	
	mxComplexDouble *thread_cutout;
	mxComplexDouble *thread_channels;
	double thread_cutout_pwr;
	double *thread_shifts;
	int thread_shiftPts;
	int thread_fftlen;
	int thread_numChans;
	int thread_chanLength;
	
	// IPP DFT vars
	Ipp8u *thread_pDFTBuffer;
	IppsDFTSpec_C_64fc *thread_pDFTSpec;
	
	double *thread_allproductpeaks;
	int *thread_allfreqlist_inds;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

void cumsum(double *in, double *out, int length){
	double value = 0;
	for (int i = 0; i<length; i++){
		value = value + in[i];
		out[i] = value;
	}
}

// double abs_fftwcomplex(fftw_complex in){
	// double val = sqrt(in[0]*in[0] + in[1]*in[1]);
	// return val;
// }

unsigned __stdcall xcorr_multichannel(void *pArgs){
    struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	mxComplexDouble *cutout = inner_data->thread_cutout;
	mxComplexDouble *channels = inner_data->thread_channels;
	double cutout_pwr = inner_data->thread_cutout_pwr;
	double *shifts = inner_data->thread_shifts;
	int shiftPts = inner_data->thread_shiftPts;
	int fftlen = inner_data->thread_fftlen;
	int numChans = inner_data->thread_numChans;
	int chanLength = inner_data->thread_chanLength;
	// IPP DFT vars
	Ipp8u *pDFTBuffer = inner_data->thread_pDFTBuffer;
	IppsDFTSpec_C_64fc *pDFTSpec = inner_data->thread_pDFTSpec;
	// outputs
	double *allproductpeaks = inner_data->thread_allproductpeaks;
	int *allfreqlist_inds = inner_data->thread_allfreqlist_inds;
	
    int i, k; // declare to simulate threads later
	double maxval, y_pwr;
	int maxind, curr_shift;
	
	// new thread vars for multi-channel process
	// int chnl_startIdx;
	Ipp64f *powerSpectrum, *power_cumu;
	int downsamplePhase;

	powerSpectrum = (Ipp64f*)ippsMalloc_64f_L(chanLength);
	power_cumu = (Ipp64f*)ippsMalloc_64f_L(chanLength);
	Ipp64fc *y = (Ipp64fc*)ippsMalloc_64fc_L(chanLength);
	Ipp64fc *dft_in = (Ipp64fc*)ippsMalloc_64fc_L(fftlen);
	Ipp64fc *dft_out = (Ipp64fc*)ippsMalloc_64fc_L(fftlen);
    
	// pick CHANNEL based on thread number
	for (k = t_ID; k<numChans; k = k+NUM_THREADS){
		downsamplePhase = k; // the downsample phase is the same as the channel number
		ippsSampleDown_64fc((Ipp64fc*)channels, numChans * chanLength, y, &chanLength, numChans, &downsamplePhase); // save the channel in y
		
		curr_shift = (int)shifts[0] - 1;
		// calculate power spectrum for whole channel
		ippsPowerSpectr_64fc(y, powerSpectrum, chanLength);
		cumsum(powerSpectrum,power_cumu,chanLength);
		
		// // run through all the shiftPts
		for (i = 0; i<shiftPts; i++){
			curr_shift = (int)shifts[i]-1;
			if (curr_shift == 0){ y_pwr = power_cumu[fftlen-1];} // unlikely to ever happen, but whatever
			else{ y_pwr = power_cumu[curr_shift + fftlen - 1] - power_cumu[curr_shift - 1];}

			ippsMul_64fc((Ipp64fc*)cutout,(Ipp64fc*)&y[curr_shift], (Ipp64fc*)dft_in, fftlen);
		
			ippsDFTFwd_CToC_64fc(dft_in, dft_out, pDFTSpec, pDFTBuffer);
		
			ippsPowerSpectr_64fc(dft_out, powerSpectrum, fftlen); // i don't need powerspectrum any more after getting power_cumu any more so use it to store the powerSpectrum of fft
		
			ippsMaxIndx_64f(powerSpectrum, fftlen, &maxval, &maxind); // it doesn't matter that powerSpectrum is longer than fftlen anyway...
			
			allproductpeaks[k*shiftPts+i] = maxval/cutout_pwr/y_pwr;
			allfreqlist_inds[k*shiftPts+i] = maxind;
		}
	}
	
	ippsFree(y);
	ippsFree(dft_in); ippsFree(dft_out);
	ippsFree(power_cumu);
	ippsFree(powerSpectrum);
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
	mxComplexDouble *cutout, *channels;
    double *shifts; // direct matlab inputs are always in doubles
	double cutout_pwr;
	int	shiftPts, numChans, chanLength;
	int fftlen;
	// declare outputs
	double *allproductpeaks;
	int *allfreqlist_inds; // declared to be int below
    
	// //reserve stuff for threads
    int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
    }

    cutout = mxGetComplexDoubles(prhs[0]); 
	channels = mxGetComplexDoubles(prhs[1]); 
	numChans = (int)mxGetScalar(prhs[2]);
	cutout_pwr = mxGetScalar(prhs[3]);
	shifts = mxGetDoubles(prhs[4]);
    
    fftlen = (int)mxGetM(prhs[0])*(int)mxGetN(prhs[0]); // this is the length of the fft, assume its only 1-D so we just take the product
	
	shiftPts = (int)mxGetN(prhs[4]); // this is the number of shifts there are
	
	/* check for proper orientation of channels */
    if (numChans != (int)mxGetM(prhs[1])){
        mexErrMsgTxt("numChans does not match number of rows of channels! Make sure that each channel is aligned in rows i.e. for n channels there must be n rows; probably just invoke .' at the end.");
    }
    
    /* if everything is fine, get the total channel length too */
    chanLength = (int)mxGetN(prhs[1]);
	
	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((int)(shiftPts),(int)(numChans),mxREAL);
    plhs[1] = mxCreateNumericMatrix((int)(shiftPts),(int)(numChans), mxINT32_CLASS,mxREAL); //initializes to 0
    
    /* get a pointer to the real data in the output matrix */
    allproductpeaks = mxGetDoubles(plhs[0]);
    allfreqlist_inds = (int*)mxGetInt32s(plhs[1]);
    
	
	// ===== IPP DFT Allocations =====
	start_t = GetCounter();
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;   
	ippsDFTGetSize_C_64fc(fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf); // this just fills the 3 integers
	/* memory allocation */
	IppsDFTSpec_C_64fc **pSpec = (IppsDFTSpec_C_64fc**)ippMalloc(sizeof(IppsDFTSpec_C_64fc*)*NUM_THREADS);
	Ipp8u **pBuffer = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	Ipp8u **pMemInit = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	for (t = 0; t<NUM_THREADS; t++){ // make one for each thread
		pSpec[t] = (IppsDFTSpec_C_64fc*)ippMalloc(sizeSpec); // this is analogue of the fftw plan
		pBuffer[t] = (Ipp8u*)ippMalloc(sizeBuf);
		pMemInit[t] = (Ipp8u*)ippMalloc(sizeInit);
		ippsDFTInit_C_64fc(fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,  pSpec[t], pMemInit[t]); // kinda like making the fftw plan?
	}
	end_t = GetCounter();
	totalTime = (end_t - start_t)/PCFreq; // in ms
	printf("Time to prepare IPP DFTs = %g ms \n",totalTime);

    // =============/* call the computational routine */==============
	GROUP_AFFINITY currentGroupAffinity, newGroupAffinity;
	start_t = GetCounter();
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_cutout = cutout;
		t_data_array[t].thread_channels = channels;

		t_data_array[t].thread_cutout_pwr = cutout_pwr;
		t_data_array[t].thread_shifts = shifts;
		t_data_array[t].thread_shiftPts = shiftPts;
		t_data_array[t].thread_fftlen = fftlen;
		t_data_array[t].thread_numChans = numChans;
		t_data_array[t].thread_chanLength = chanLength;
			
		// IPP DFT vars
		t_data_array[t].thread_pDFTBuffer = pBuffer[t];
		t_data_array[t].thread_pDFTSpec = pSpec[t];
		
		t_data_array[t].thread_allproductpeaks = allproductpeaks;
		t_data_array[t].thread_allfreqlist_inds = allfreqlist_inds;
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&xcorr_multichannel,(void*)&t_data_array[t],0,NULL);
		GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
		newGroupAffinity = currentGroupAffinity;
		newGroupAffinity.Group = t%2;
		SetThreadGroupAffinity(ThreadList[t], &newGroupAffinity, NULL);
		GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
        printf("Beginning threadID %i..\n",t_data_array[t].thread_t_ID);
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