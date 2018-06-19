#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"
// #include <immintrin.h> // include if trying the intrinsics code

#define NUM_THREADS 24 // seems like a good number now, you may increase if your computation time takes more than 1 second, otherwise spawning more threads takes longer than the actual execution time per thread lol

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
	mxComplexDouble *thread_y;
	double *thread_power_cumu;
	double thread_cutout_pwr;
	double *thread_shifts;
	int thread_shiftPts;
	int thread_fftlen;
	
	// IPP DFT vars
	Ipp8u *thread_pBuffer;
	IppsDFTSpec_C_64fc *thread_pSpec;
	
	double *thread_productpeaks;
	int *thread_freqlist_inds;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

unsigned __stdcall xcorr_singleChannel(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	mxComplexDouble *cutout = inner_data->thread_cutout;
	mxComplexDouble *y = inner_data->thread_y;
	double *power_cumu = inner_data->thread_power_cumu;
	double cutout_pwr = inner_data->thread_cutout_pwr;
	double *shifts = inner_data->thread_shifts;
	int shiftPts = inner_data->thread_shiftPts;
	int fftlen = inner_data->thread_fftlen;
	
	// IPP DFT vars
	Ipp8u *pBuffer = inner_data->thread_pBuffer;
	IppsDFTSpec_C_64fc *pSpec = inner_data->thread_pSpec;
	
	double *productpeaks = inner_data->thread_productpeaks;
	int *freqlist_inds = inner_data->thread_freqlist_inds;
	// end of attached variables

	// computations
	int i, curr_shift;
	double y_pwr;
	Ipp64fc *dft_in = (Ipp64fc*)ippsMalloc_64fc_L(fftlen);
	Ipp64fc *dft_out = (Ipp64fc*)ippsMalloc_64fc_L(fftlen);
	Ipp64f *magnSq = (Ipp64f*)ippsMalloc_64f_L(fftlen);
	Ipp64f maxval;
	int maxind;
	
	// pick point based on thread number
	for (i = t_ID; i<shiftPts; i=i+NUM_THREADS){
		curr_shift = (int)shifts[i]-1;
		// printf("Working on thread %i, loop %i, shift %i \n",thread,i,curr_shift);
		if (curr_shift == 0){ y_pwr = power_cumu[fftlen-1];} // unlikely to ever happen, but whatever
		else{ y_pwr = power_cumu[curr_shift + fftlen - 1] - power_cumu[curr_shift - 1];}
		
        ippsMul_64fc((Ipp64fc*)cutout,(Ipp64fc*)&y[curr_shift], (Ipp64fc*)dft_in, fftlen);
		
		ippsDFTFwd_CToC_64fc(dft_in, dft_out, pSpec, pBuffer);
		
		ippsPowerSpectr_64fc(dft_out, magnSq, fftlen);
		
		ippsMaxIndx_64f(magnSq, fftlen, &maxval, &maxind);
		
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
    mxComplexDouble *cutout, *y;
    double *power_cumu, *shifts; // direct matlab inputs are always in doubles
	double cutout_pwr;
	int	shiftPts;
	int m, fftlen;
	// declare outputs
	double *productpeaks;
	int *freqlist_inds; // declared to be int below

	// //reserve stuff for threads
    int t; 
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
    }

    cutout = mxGetComplexDoubles(prhs[0]); 
	y = mxGetComplexDoubles(prhs[1]); 
	power_cumu = mxGetDoubles(prhs[2]);
	cutout_pwr = mxGetScalar(prhs[3]);
	shifts = mxGetDoubles(prhs[4]);
    
    m = (int)mxGetM(prhs[0]);
    fftlen = m*(int)mxGetN(prhs[0]); // this is the length of the fft, assume its only 1-D so we just take the product
	
	shiftPts = (int)mxGetN(prhs[4]); // this is the number of shifts there are
	
	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,(int)shiftPts,mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, (int)shiftPts, mxINT32_CLASS,mxREAL); //initializes to 0
    
    /* get a pointer to the real data in the output matrix */
    productpeaks = mxGetDoubles(plhs[0]);
    freqlist_inds = (int*)mxGetInt32s(plhs[1]);
	
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
	// ================================================================
	start_t = GetCounter();

    // =============/* call the computational routine */==============
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
		
		t_data_array[t].thread_productpeaks = productpeaks;
		t_data_array[t].thread_freqlist_inds = freqlist_inds;
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&xcorr_singleChannel,(void*)&t_data_array[t],0,NULL);
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


// void dotstar_splitsplit2fftw(double *r_x, double *i_x, double *r_y, double *i_y, fftw_complex *out, int len){
	// int i;
    // double A, B, C;
	// for (i=0;i<len;i++){
        // A = i_x[i] * i_y[i];
        // B = r_x[i] * r_y[i];
        // out[i][0] = B - A;
        // C = (r_x[i] + i_x[i])*(r_y[i] + i_y[i]);
        // out[i][1] = C - B - A;
// // 		out[i][0] = r_x[i]*r_y[i] - i_x[i]*i_y[i]; // old code with 4 mults
// // 		out[i][1] = r_x[i]*i_y[i] + r_y[i]*i_x[i];
	// }
// // //     BELOW IS THE INTRINSICS CODE, NO APPRECIABLE SPEED INCREASE (~3% at most), WORKS THO (BUT DOES NOT ACCOUNT FOR NON FACTOR OF 4 ARRAY LENGTHS)
// //     int i;
// //     double *dout, *dout_i;
// //     __m256d xr, xi, yr, yi, outr, outi, A, B, C;
// //     for (i=0;i<len;i=i+4){
// //         xr = _mm256_loadu_pd(&r_x[i]);
// //         xi = _mm256_loadu_pd(&i_x[i]);
// //         yr = _mm256_loadu_pd(&r_y[i]);
// //         yi = _mm256_loadu_pd(&i_y[i]);
// //         
// //         A = _mm256_mul_pd(xi,yi); // this is bd
// //         outr = _mm256_fmsub_pd(xr,yr,A);
// //         dout = (double*)&outr;
// //         out[i][0] = dout[0]; // copy to fftw_complex, real parts
// //         out[i+1][0] = dout[1];
// //         out[i+2][0] = dout[2];
// //         out[i+3][0] = dout[3];
// //         
// //         B = _mm256_add_pd(xr,xi);
// //         C = _mm256_add_pd(yr,yi);
// //         outi = _mm256_fmsub_pd(B,C,outr); // its now (a+b)(c+d) - ac + bd
// //         outi = _mm256_sub_pd(outi,A);
// //         outi = _mm256_sub_pd(outi,A); // now its (a+b)(c+d) - ac - bd, made 2 subtractions
// //         dout_i = (double*)&outi;
// //         out[i][1] = dout_i[0]; // copy to fftw_complex, imag parts
// //         out[i+1][1] = dout_i[1];
// //         out[i+2][1] = dout_i[2];
// //         out[i+3][1] = dout_i[3];
// //     }
// }