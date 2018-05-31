#include "mex.h"
#include <math.h>
#include "stdio.h"
#include "fftw3.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"
// #include <immintrin.h> // include if trying the intrinsics code


// #include "blas.h"

// #if !defined(_WIN32)
// #define ddot ddot_
// #define dnrm2 dnrm2_
// #define dzasum dzasum_
// #endif

static volatile int WaitForThread[24];
const int nThreads = 24;
// test fftw_plans array on stack for threads, works
fftw_plan allplans[24]; // REMEMBER TO CHECK FFTW PLANS CREATION IN THE ENTRY FUNCTION

void dotstar_splitsplit2fftw(double *r_x, double *i_x, double *r_y, double *i_y, fftw_complex *out, int len){
	int i;
    double A, B, C;
	for (i=0;i<len;i++){
        A = i_x[i] * i_y[i];
        B = r_x[i] * r_y[i];
        out[i][0] = B - A;
        C = (r_x[i] + i_x[i])*(r_y[i] + i_y[i]);
        out[i][1] = C - B - A;
// 		out[i][0] = r_x[i]*r_y[i] - i_x[i]*i_y[i]; // old code with 4 mults
// 		out[i][1] = r_x[i]*i_y[i] + r_y[i]*i_x[i];
	}
// //     BELOW IS THE INTRINSICS CODE, NO APPRECIABLE SPEED INCREASE (~3% at most), WORKS THO (BUT DOES NOT ACCOUNT FOR NON FACTOR OF 4 ARRAY LENGTHS)
//     int i;
//     double *dout, *dout_i;
//     __m256d xr, xi, yr, yi, outr, outi, A, B, C;
//     for (i=0;i<len;i=i+4){
//         xr = _mm256_loadu_pd(&r_x[i]);
//         xi = _mm256_loadu_pd(&i_x[i]);
//         yr = _mm256_loadu_pd(&r_y[i]);
//         yi = _mm256_loadu_pd(&i_y[i]);
//         
//         A = _mm256_mul_pd(xi,yi); // this is bd
//         outr = _mm256_fmsub_pd(xr,yr,A);
//         dout = (double*)&outr;
//         out[i][0] = dout[0]; // copy to fftw_complex, real parts
//         out[i+1][0] = dout[1];
//         out[i+2][0] = dout[2];
//         out[i+3][0] = dout[3];
//         
//         B = _mm256_add_pd(xr,xi);
//         C = _mm256_add_pd(yr,yi);
//         outi = _mm256_fmsub_pd(B,C,outr); // its now (a+b)(c+d) - ac + bd
//         outi = _mm256_sub_pd(outi,A);
//         outi = _mm256_sub_pd(outi,A); // now its (a+b)(c+d) - ac - bd, made 2 subtractions
//         dout_i = (double*)&outi;
//         out[i][1] = dout_i[0]; // copy to fftw_complex, imag parts
//         out[i+1][1] = dout_i[1];
//         out[i+2][1] = dout_i[2];
//         out[i+3][1] = dout_i[3];
//     }
}


double abs_fftwcomplex(fftw_complex in){
	double val = sqrt(in[0]*in[0] + in[1]*in[1]);
	return val;
}

unsigned __stdcall myfunc(void *pArgs){
    void **Args = (void**)pArgs;
	//declare inputs
    mxComplexDouble *cutout, *y;
	double *power_cumu, *shifts;
	double *cutout_pwr_ptr, cutout_pwr;
	int *shiftPts_ptr, shiftPts, *fftlen_ptr, fftlen;
	fftw_complex *fin, *fout;
	//declare outputs
	double *productpeaks;
	int *freqlist_inds;
	
    int i, k; // declare to simulate threads later
	double curr_max, newmax, y_pwr;
	int maxind, curr_shift;
    
	//declare thread variables
	double *ThreadID;
	int t_ID;
	
	// assignments from Args passed in
	cutout = (mxComplexDouble*)Args[0];
// 	cutout_i = (double*)Args[1];
	y = (mxComplexDouble*)Args[2];
// 	y_i = (double*)Args[3];
	power_cumu = (double*)Args[4];
	cutout_pwr_ptr = (double*)Args[5];
	cutout_pwr = (double)cutout_pwr_ptr[0]; // this has same length as y, y_i
	shifts = (double*)Args[6];
	shiftPts_ptr = (int*)Args[7];
	shiftPts = (int)shiftPts_ptr[0];
	fftlen_ptr = (int*)Args[8];
	fftlen = (int)fftlen_ptr[0]; // length of fft i.e. length of cutout
	// fftw related
	fin = (fftw_complex*)Args[9]; // these should be initialized as size = nThreads*fftlen
	fout = (fftw_complex*)Args[10];
	// outputs
	productpeaks = (double*)Args[11];
	freqlist_inds = (int*)Args[12];
	
	// now assign threadID within function
	ThreadID = (double*)Args[13];
    t_ID = (int)ThreadID[0];
    // allow new threads to be assigned in mexFunction
    WaitForThread[t_ID]=0;

    
	// pick point based on thread number
	for (i = t_ID; i<shiftPts; i=i+nThreads){
		curr_shift = (int)shifts[i]-1;
		// printf("Working on thread %i, loop %i, shift %i \n",thread,i,curr_shift);
		if (curr_shift == 0){ y_pwr = power_cumu[fftlen-1];} // unlikely to ever happen, but whatever
		else{ y_pwr = power_cumu[curr_shift + fftlen - 1] - power_cumu[curr_shift - 1];}
		
        ippsMul_64fc((Ipp64fc*)cutout,(Ipp64fc*)&y[curr_shift], (Ipp64fc*)&fin[fftlen*t_ID], fftlen);

		fftw_execute(allplans[t_ID]);
		
		curr_max = abs_fftwcomplex(fout[fftlen*t_ID]); // the first value
		maxind = 0;
		for (k=1;k<fftlen;k++){
			newmax = abs_fftwcomplex(fout[fftlen*t_ID+k]);
			if (newmax>curr_max){curr_max = newmax; maxind = k;}
		}
		
		productpeaks[i] = curr_max*curr_max/cutout_pwr/y_pwr;
		freqlist_inds[i] = maxind;
	}
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // declare variables
    int i;
    mxComplexDouble *cutout, *y;
    double *power_cumu, *shifts; // direct matlab inputs are always in doubles
	double cutout_pwr;
	int	shiftPts;
	int m, fftlen;
	// declare outputs
	double *productpeaks;
	int *freqlist_inds; // declared to be int below
    
    fftw_complex *fin;
    fftw_complex *fout;
    
	clock_t start, end;
	
	// //reserve stuff for threads
    double *ThreadIDList;
    void **ThreadArgs;
    int t, sleep; // for loops over threads
    HANDLE *ThreadList; // handles to threads
    ThreadIDList = (double*)mxMalloc(nThreads*sizeof(double));
    ThreadList = (HANDLE*)mxMalloc(nThreads*sizeof(HANDLE));
    ThreadArgs = (void**)mxMalloc(14*sizeof(void*));
    sleep = 0;
	//assign threadIDs
    for(t=0;t<nThreads;t++){ThreadIDList[t] = t;}
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
    }

    cutout = mxGetComplexDoubles(prhs[0]); 
//     cutout_i = mxGetPi(prhs[0]);
	y = mxGetComplexDoubles(prhs[1]); 
// 	y_i = mxGetPi(prhs[1]);
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
	
    // ====== ALLOC VARS FOR FFT IN THREADS BEFORE PLANS ====================

    fin = fftw_alloc_complex(fftlen*nThreads);
    fout = fftw_alloc_complex(fftlen*nThreads);
	// ======== MAKE PLANS BEFORE COMPUTATIONS IN THREADS  ============
	start = clock();
    allplans[0] = fftw_plan_dft_1d(fftlen, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE); // FFTW_MEASURE seems to cut execution time by ~10%, but FFTW_ESTIMATE takes ~0.001s whereas MEASURE takes ~0.375s
    end = clock();
    printf("Time for 1st single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
    
	start = clock();
    for (i=1;i<nThreads;i++){
        allplans[i] = fftw_plan_dft_1d(fftlen, &fin[fftlen*i], &fout[fftlen*i], FFTW_FORWARD, FFTW_ESTIMATE); // make the other plans, not executing them yet
    }
    end = clock();
    printf("Time for 2nd-n'th single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
	// ================================================================
	// ======== ATTACH VARS TO ARGS =======================
	ThreadArgs[0] = (void*)cutout;
// 	ThreadArgs[1] = (void*)cutout_i;
	ThreadArgs[2] = (void*)y;
// 	ThreadArgs[3] = (void*)y_i;
	ThreadArgs[4] = (void*)power_cumu;
	ThreadArgs[5] = (void*)&cutout_pwr;
	ThreadArgs[6] = (void*)shifts;
	ThreadArgs[7] = (void*)&shiftPts;
	ThreadArgs[8] = (void*)&fftlen; // length of fft i.e. length of cutout
	// fftw related
	ThreadArgs[9] = (void*)fin;
	ThreadArgs[10] = (void*)fout;
	// outputs
	ThreadArgs[11] = (void*)productpeaks;
	ThreadArgs[12] = (void*)freqlist_inds;
	
	
    // =============/* call the computational routine */==============
    //start threads
    for(t=0;t<nThreads;t++){
        while (t>0 && WaitForThread[t-1]==1){sleep=sleep+1; Sleep(1); printf("Slept %i..\n",sleep);}// wait for previous threads to assign ID within function
        ThreadArgs[13] = (void*)&ThreadIDList[t]; // assign the threadID
        WaitForThread[t] = 1;
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&myfunc,(void*)ThreadArgs,0,NULL);
        printf("Beginning threadID %i..%i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    
    WaitForMultipleObjects(nThreads,ThreadList,1,INFINITE);
	
	
	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<nThreads;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    printf("All threads closed! \n");

    for (i=0;i<nThreads;i++){fftw_destroy_plan(allplans[i]);}

    fftw_free(fin);
    fftw_free(fout);
	
	mxFree(ThreadIDList);
    mxFree(ThreadList);
    mxFree(ThreadArgs);
}