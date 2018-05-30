// this is a modification of testcomplex; small bandwidth with high number of channels results in poor use of processor power
// it is better to assign each thread to then solve each channel instead?

// Call via:
// [allproductpeaks, allfreqlist_inds] = dot_xcorr(conjcutout, channels, numChans, cutout_pwr, shifts);

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include "fftw3.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
#include "ipp.h"

static volatile int WaitForThread[6];
const int nThreads = 6;
// test fftw_plans array on stack for threads, works
fftw_plan allplans[6]; // REMEMBER TO CHECK FFTW PLANS CREATION IN THE ENTRY FUNCTION

void dotstar_splitsplit2fftw(double *r_x, double *i_x, double *r_y, double *i_y, fftw_complex *out, int len){
	int i;
    double A, B, C;
	for (i=0;i<len;i++){
        A = i_x[i] * i_y[i];
        B = r_x[i] * r_y[i];
        out[i][0] = B - A;
        C = (r_x[i] + i_x[i])*(r_y[i] + i_y[i]);
        out[i][1] = C - B - A;
	}
}


// void stridedcpy2complex(fftw_complex *out, double *in_r, double *in_i, int len){ // len is length of in arrays, out length is double of that due to complex
//     int i;
//     for (i=0;i<len;i++){ // unrolling loops does not seem to help
//         out[i][0] = in_r[i];
//         out[i][1] = in_i[i];
//     }
// }
// 
// void stridedcpy2ri(fftw_complex *in, double *out_r, double *out_i, int len){
//     int i;
//     for (i=0;i<len;i++){
//         out_r[i] = in[i][0];
//         out_i[i] = in[i][1];
//     }
// }

double abs_fftwcomplex(fftw_complex in){
	double val = sqrt(in[0]*in[0] + in[1]*in[1]);
	return val;
}

unsigned __stdcall myfunc(void *pArgs){
    void **Args = (void**)pArgs;
	//declare inputs
	double *cutout, *cutout_i, *channels, *channels_i, *shifts;
	double *cutout_pwr_ptr, cutout_pwr;
	int *shiftPts_ptr, shiftPts, *fftlen_ptr, fftlen, *numChans_ptr, numChans, *chanLength_ptr, chanLength;
	fftw_complex *fin, *fout;
	//declare outputs
	double *allproductpeaks;
	int *allfreqlist_inds;
	
    int i, j, k; // declare to simulate threads later
	double curr_max, newmax, y_pwr;
	int maxind, curr_shift;
    
	//declare thread variables
	double *ThreadID;
	int t_ID;
	
	// new thread vars for multi-channel process
	int chnl_startIdx;
	Ipp64f realNorm, imagNorm;
	
	// assignments from Args passed in
	cutout = (double*)Args[0];
	cutout_i = (double*)Args[1];
	channels = (double*)Args[2];
	channels_i = (double*)Args[3];
	numChans_ptr = (int*)Args[4];
	numChans = (int)numChans_ptr[0];
	chanLength_ptr = (int*)Args[5];
	chanLength = (int)chanLength_ptr[0];
	cutout_pwr_ptr = (double*)Args[6];
	cutout_pwr = (double)cutout_pwr_ptr[0]; // this has same length as y, y_i
	shifts = (double*)Args[7];
	shiftPts_ptr = (int*)Args[8];
	shiftPts = (int)shiftPts_ptr[0];
	fftlen_ptr = (int*)Args[9];
	fftlen = (int)fftlen_ptr[0]; // length of fft i.e. length of cutout
	// fftw related
	fin = (fftw_complex*)Args[10]; // these should be initialized as size = nThreads*fftlen
	fout = (fftw_complex*)Args[11];
	// outputs
	allproductpeaks = (double*)Args[12];
	allfreqlist_inds = (int*)Args[13];
	
	// now assign threadID within function
	ThreadID = (double*)Args[14];
    t_ID = (int)ThreadID[0];
    // allow new threads to be assigned in mexFunction
    WaitForThread[t_ID]=0;

    
	// pick CHANNEL based on thread number
	for (k = t_ID; k<numChans; k = k+nThreads){
		chnl_startIdx = k*chanLength;
		curr_shift = (int)shifts[0] - 1;
		ippsNorm_L2_64f(&channels[chnl_startIdx + curr_shift], fftlen, &realNorm);
		ippsNorm_L2_64f(&channels_i[chnl_startIdx + curr_shift], fftlen, &imagNorm); // basically calculate the norms for the first shift directly
		y_pwr = realNorm*realNorm + imagNorm*imagNorm;
		
		// // run through all the shiftPts
		for (i = 0; i<shiftPts; i++){
			curr_shift = (int)shifts[i]-1;
		
			// for every loop after the first, adjust the y_pwr by adding next sample and deducting first
			if (i>0){
				y_pwr = y_pwr + channels[chnl_startIdx + curr_shift + fftlen - 1]*channels[chnl_startIdx + curr_shift + fftlen - 1] + channels_i[chnl_startIdx + curr_shift + fftlen - 1]*channels_i[chnl_startIdx + curr_shift + fftlen - 1]; // first we add the new sample energy
				y_pwr = y_pwr - (channels[chnl_startIdx + curr_shift - 1]*channels[chnl_startIdx + curr_shift - 1]) - (channels_i[chnl_startIdx + curr_shift - 1]*channels_i[chnl_startIdx + curr_shift - 1]); // then deduct the sample we just left behind
			}

            dotstar_splitsplit2fftw(cutout, cutout_i, &channels[chnl_startIdx + curr_shift], &channels_i[chnl_startIdx + curr_shift], &fin[fftlen*t_ID], fftlen);

            fftw_execute(allplans[t_ID]);
		
			curr_max = abs_fftwcomplex(fout[fftlen*t_ID]); // the first value
			maxind = 0;
			for (j=1;j<fftlen;j++){
				newmax = abs_fftwcomplex(fout[fftlen*t_ID+j]);
				if (newmax>curr_max){curr_max = newmax; maxind = j;}
			}

			allproductpeaks[k*shiftPts+i] = curr_max*curr_max/cutout_pwr/y_pwr;
			allfreqlist_inds[k*shiftPts+i] = maxind;
		}
	}
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // declare variables
    int i;
    double *cutout, *cutout_i, *channels, *channels_i, *shifts; // direct matlab inputs are always in doubles
	double cutout_pwr;
	int	shiftPts, numChans, chanLength;
	int m, fftlen;
	// declare outputs
	double *allproductpeaks;
	int *allfreqlist_inds; // declared to be int below
    
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
    ThreadArgs = (void**)mxMalloc(15*sizeof(void*));
    sleep = 0;
	//assign threadIDs
    for(t=0;t<nThreads;t++){ThreadIDList[t] = t;}
    
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
    }

    cutout = mxGetPr(prhs[0]); 
    cutout_i = mxGetPi(prhs[0]);
	channels = mxGetPr(prhs[1]); 
	channels_i = mxGetPi(prhs[1]);
	numChans = (int)mxGetScalar(prhs[2]);
	cutout_pwr = mxGetScalar(prhs[3]);
	shifts = mxGetPr(prhs[4]);
    
    m = (int)mxGetM(prhs[0]);
    fftlen = m*(int)mxGetN(prhs[0]); // this is the length of the fft, assume its only 1-D so we just take the product
	
	shiftPts = (int)mxGetN(prhs[4]); // this is the number of shifts there are
	
	/* check for proper orientation of channels */
    if (numChans != (int)mxGetN(prhs[1])){
        mexErrMsgTxt("numChans does not match number of columns of channels! Make sure that each channel is aligned in columns i.e. for n channels there must be n columns; probably just invoke .' at the end.");
    }
    
    /* if everything is fine, get the total channel length too */
    chanLength = (int)mxGetM(prhs[1]);
	
	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)(shiftPts),(mwSize)(numChans),mxREAL);
    plhs[1] = mxCreateNumericMatrix((mwSize)(shiftPts),(mwSize)(numChans), mxINT32_CLASS,mxREAL); //initializes to 0
    
    /* get a pointer to the real data in the output matrix */
    allproductpeaks = mxGetPr(plhs[0]);
    allfreqlist_inds = (int*)mxGetPr(plhs[1]);
    
	
    // ====== ALLOC VARS FOR FFT IN THREADS BEFORE PLANS ====================

    fin = fftw_alloc_complex(fftlen*nThreads);
    fout = fftw_alloc_complex(fftlen*nThreads);
	// ======== MAKE PLANS BEFORE COMPUTATIONS IN THREADS  ============
	start = clock();
    allplans[0] = fftw_plan_dft_1d(fftlen, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT); // FFTW_MEASURE seems to cut execution time by ~10%, but FFTW_ESTIMATE takes ~0.001s whereas MEASURE takes ~0.375s
    end = clock();
    printf("Time for 1st single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
    
	start = clock();
    for (i=1;i<nThreads;i++){
        allplans[i] = fftw_plan_dft_1d(fftlen, &fin[fftlen*i], &fout[fftlen*i], FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT); // make the other plans, not executing them yet
    }
    end = clock();
    printf("Time for 2nd-n'th single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
	// ================================================================
	// ======== ATTACH VARS TO ARGS =======================
	ThreadArgs[0] = (void*)cutout;
	ThreadArgs[1] = (void*)cutout_i;
	ThreadArgs[2] = (void*)channels;
	ThreadArgs[3] = (void*)channels_i;
	ThreadArgs[4] = (void*)&numChans;
	ThreadArgs[5] = (void*)&chanLength;
	ThreadArgs[6] = (void*)&cutout_pwr;
	ThreadArgs[7] = (void*)shifts;
	ThreadArgs[8] = (void*)&shiftPts;
	ThreadArgs[9] = (void*)&fftlen; // length of fft i.e. length of cutout
	// fftw related
	ThreadArgs[10] = (void*)fin;
	ThreadArgs[11] = (void*)fout;
	// outputs
	ThreadArgs[12] = (void*)allproductpeaks;
	ThreadArgs[13] = (void*)allfreqlist_inds;
	
	
    // =============/* call the computational routine */==============
	ippInit();
    //start threads
    for(t=0;t<nThreads;t++){
        while (t>0 && WaitForThread[t-1]==1){sleep=sleep+1; Sleep(1); printf("Slept %i..\n",sleep);}// wait for previous threads to assign ID within function
        ThreadArgs[14] = (void*)&ThreadIDList[t]; // assign the threadID
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

// // NO POINT IN USING THIS, WOULD HAVE TO DO .* FIRST ANYWAY
// void forwardShiftEleComplex(double *in_r_cut, double *in_i_cut, fftw_complex *curr, int shift, int len){ // in_r_cut and in_i_cut should only be 'shift' elements long
    // int i;
	// memmove(&curr[0],&curr[shift],(len-shift)*sizeof(fftw_complex)); // shifts the elements from index 'shift' onwards forward towards the first index
	// // for (i=0;i<len-shift;i++){ // manual memmove of above, this is slower at the 10k length array benchmark at least
		// // curr[i][0] = curr[i+1][0];
		// // curr[i][1] = curr[i+1][1];
	// // }
    // for (i=0;i<shift;i++){
        // curr[i+len-shift][0] = in_r_cut[i];
		// curr[i+len-shift][1] = in_i_cut[i];
    // }
// }
