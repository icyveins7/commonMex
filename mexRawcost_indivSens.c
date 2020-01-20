#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"
// #include <immintrin.h> // include if trying the intrinsics code

#define LIGHT_SPD 299792458.0
#define NUM_THREADS 4 // seems like a good number now, you may increase if your computation time takes more than 1 second, otherwise spawning more threads takes longer than the actual execution time per thread lol

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
	
	double *thread_td_m;
	double *thread_sig_m;
	int thread_numSpacePts;
	int thread_num_td;
	double *thread_space_pos; 
	double *thread_sens_pos;
	
	double *thread_costs;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

unsigned __stdcall rawcost_thread(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	double *td_m = inner_data->thread_td_m;
	double *sig_m = inner_data->thread_sig_m;
	int numSpacePts = inner_data->thread_numSpacePts;
	int num_td = inner_data->thread_num_td:
	double *space_pos = inner_data->thread_space_pos;
	double *sens_pos = inner_data->thread_sens_pos;
	
	double *costs = inner_data->thread_costs;
	// end of attached variables

	// computations
	int i;
	int k;
	double t_0[3];
	double t_1[3]; td_t;
	double norm_t0, norm_t1;
	
	
	// pick point based on thread number
	for (i = t_ID; i<numSpacePts; i=i+NUM_THREADS){
		costs[i] = 0;
		for (k = 0; k < num_td; k++){
			ippsSub_64f(&sens_pos[i*3*2 + 0*3], &space_pos[i*3], t_0, 3);
			ippsSub_64f(&sens_pos[i*3*2 + 1*3], &space_pos[i*3], t_1, 3);
			ippsNorm_L2_64f(t_0, 3, &norm_t0);
			ippsNorm_L2_64f(t_1, 3, &norm_t1);
			
			td_t = (norm_t1 - norm_t0)/LIGHT_SPD;
			costs[i] = costs[i] + ( (td_t - td_m[k]) * (td_t - td_m[k]) / sig_m[k] / sig_m[k] );
	}
	
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
	// start_t = GetCounter();
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
	// end_t = GetCounter();
	// totalTime = (end_t - start_t)/PCFreq; // in ms
	// printf("Time to prepare IPP DFTs = %g ms \n",totalTime);
	// ================================================================
	// start_t = GetCounter();

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
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&rawcost_thread,(void*)&t_data_array[t],0,NULL);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
	
	// ============== CLEANUP =================
    // close threads
    // printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    // printf("All threads closed! \n");
	// end_t = GetCounter();
	// totalTime = (end_t - start_t)/PCFreq; // in ms
	// printf("Time for threads to finish = %g ms \n",totalTime);
	
	for (t=0; t<NUM_THREADS; t++){
		ippFree(pSpec[t]);
		ippFree(pBuffer[t]);
		ippFree(pMemInit[t]);
	}
	ippFree(pSpec);
	ippFree(pBuffer);
	ippFree(pMemInit);
}
