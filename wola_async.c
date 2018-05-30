/*
 * Performs WOLA FFT to channelize data.
 *
 * testwola(y,f_tap,fftlen,Dec)
 * Inputs: signal, filter tap coeffs, number of FFT bins, decimation factor
 *
*/

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include "fftw3.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>


// #include "blas.h"

// #if !defined(_WIN32)
// #define ddot ddot_
// #define dnrm2 dnrm2_
// #define dzasum dzasum_
// #endif

static volatile int WaitForThread[6]; // only have 20 channels, increase if changed
const int nThreads = 6;
// test fftw_plans array on stack for threads, works
fftw_plan allplans[6]; // REMEMBER TO CHECK FFTW PLANS CREATION IN THE ENTRY FUNCTION


// void stridedcpy2complex(fftw_complex *out, double *in_r, double *in_i, int len){ // len is length of in arrays, out length is double of that due to complex
//     int i;
//     for (i=0;i<len;i++){ // unrolling loops does not seem to help
//         out[i][0] = in_r[i];
//         out[i][1] = in_i[i];
//     }
// }

void stridedcpy2ri(fftw_complex *in, double *out_r, double *out_i, int len){
    int i;
    for (i=0;i<len;i++){
        out_r[i] = in[i][0];
        out_i[i] = in[i][1];
    }
}

// double abs_fftwcomplex(fftw_complex in){
// 	double val = sqrt(in[0]*in[0] + in[1]*in[1]);
// 	return val;
// }

unsigned __stdcall myfunc(void *pArgs){
    void **Args = (void**)pArgs;
	//declare inputs
	double *y, *y_i, *f_tap;
	mwSize *nprimePts_ptr, nprimePts, *N_ptr, N; // rename fftlen to N
    mwSize *Dec_ptr, Dec, *L_ptr, L;
	fftw_complex *fin, *fout;
	//declare outputs
	double *out_r, *out_i;
    
//     double testacc;
	
	//declare thread variables
	double *ThreadID;
	int t_ID;
    
    mwSize nprime, n, a, b, offset; // declare to simulate threads later
    mwSize k;
	
	// assignments from Args passed in
    y = (double*)Args[0];
    y_i = (double*)Args[1];
    f_tap = (double*)Args[2];
    N_ptr = (mwSize*)Args[3];
    N = N_ptr[0]; // length of fft
    Dec_ptr = (mwSize*)Args[4];
    Dec = Dec_ptr[0];
    L_ptr = (mwSize*)Args[5]; 
    L = L_ptr[0]; // no. of filter taps
    nprimePts_ptr = (mwSize*)Args[6];
    nprimePts = nprimePts_ptr[0];
    
	// fftw related
	fin = (fftw_complex*)Args[7]; // these should be initialized as size = nThreads*fftlen
	fout = (fftw_complex*)Args[8];
	// outputs
    out_r = (double*)Args[9];
    out_i = (double*)Args[10];
	
	// now assign threadID within function
	ThreadID = (double*)Args[11];
    t_ID = (int)ThreadID[0];
    // allow new threads to be assigned in mexFunction
    WaitForThread[t_ID]=0;

    
	// pick point based on thread number

    offset = t_ID * N; // used to only fill the part of the fftw_complex used for this thread

	for (nprime = t_ID; nprime<nprimePts; nprime=nprime+nThreads){
        n = nprime*Dec;
        for (a = 0; a<N; a++){
            fin[a + offset][0] = 0; // init to 0
            fin[a + offset][1] = 0;
            for (b = 0; b<L/N; b++){
                if (n - (b*N+a) >= 0){
                    fin[a + offset][0] = fin[a + offset][0] + y[n-(b*N+a)]*f_tap[b*N+a];
                    fin[a + offset][1] = fin[a + offset][1] + y_i[n-(b*N+a)]*f_tap[b*N+a]; // f_tap coeffs are all real
                } // fin is fftw_complex
            }
        }
        fftw_execute(allplans[t_ID]); // this should place them into another fftw_complex fout
        
		if (Dec*2 == N && nprime % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
			for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
				fout[offset + k][0] = -fout[offset + k][0];
				fout[offset + k][1] = -fout[offset + k][1];
			}
		}
		
        stridedcpy2ri(&fout[t_ID*N], &out_r[nprime*N], &out_i[nprime*N], N); // NOT SURE, CHECK THIS, MATLAB IS COLUMN MAJOR
	}
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // declare variables
    mwSize i;
    double  *y, *y_i, *f_tap; // direct matlab inputs are always in doubles
	mwSize nprimePts;
	mwSize fftlen, siglen, L, Dec;
	// declare outputs
	double *out_r, *out_i;
    
	clock_t start, end;
    // for fftw stuff
    fftw_complex *fin;
    fftw_complex *fout;
	
	// //reserve stuff for threads
    double *ThreadIDList;
    void **ThreadArgs;
    int t, sleep; // for loops over threads
    HANDLE *ThreadList; // handles to threads
    ThreadIDList = (double*)mxMalloc(nThreads*sizeof(double));
    ThreadList = (HANDLE*)mxMalloc(nThreads*sizeof(HANDLE));
    ThreadArgs = (void**)mxMalloc(12*sizeof(void*));
    sleep = 0;
	//assign threadIDs
    for(t=0;t<nThreads;t++){ThreadIDList[t] = t;}
    
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	y = mxGetPr(prhs[0]); 
	y_i = mxGetPi(prhs[0]);
    f_tap = mxGetPr(prhs[1]); 
	fftlen = (mwSize)mxGetScalar(prhs[2]); // this is the length of the fft i.e. number of channels
    Dec = (mwSize)mxGetScalar(prhs[3]); // decimation factor
    
    siglen = (mwSize)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
    L = (mwSize)(mxGetM(prhs[1]) * mxGetN(prhs[1])); // no. of filter taps
	
	/* argument checks */
    if (Dec != fftlen && Dec*2 != fftlen){
        mexErrMsgTxt("PHASE CORRECTION ONLY IMPLEMENTED FOR DECIMATION = FFT LENGTH OR DECIMATION * 2 = FFT LENGTH!");
    }
	if (L%fftlen != 0){
        mexErrMsgTxt("Filter taps length must be factor multiple of fft length!");
    }
	
	nprimePts = (mwSize)(siglen/Dec);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix((mwSize)fftlen,(mwSize)(nprimePts),mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out_r = mxGetPr(plhs[0]); // CHECK GET PR OR GET PI BASED ON WHAT YOU ALLOCATED
    out_i = mxGetPi(plhs[0]); // THIS IS PI
    
//     /* create the output matrix ===== TEST WITH 2XREAL DATA*/
//     plhs[0] = mxCreateDoubleMatrix(1,(mwSize)(fftlen*nprimePts),mxREAL);
//     plhs[1] = mxCreateDoubleMatrix(1,(mwSize)(fftlen*nprimePts),mxREAL);
// 
//     /* get a pointer to the real data in the output matrix */
//     out_r = mxGetPr(plhs[0]); // CHECK GET PR OR GET PI BASED ON WHAT YOU ALLOCATED
//     out_i = mxGetPr(plhs[1]); // BOTH PR

	
    // ====== ALLOC VARS FOR FFT IN THREADS BEFORE PLANS ====================

    fin = fftw_alloc_complex(fftlen*nThreads);
    fout = fftw_alloc_complex(fftlen*nThreads);
	// ======== MAKE PLANS BEFORE COMPUTATIONS IN THREADS  ============
	start = clock();
    allplans[0] = fftw_plan_dft_1d(fftlen, fin, fout, FFTW_BACKWARD, FFTW_MEASURE); // FFTW_MEASURE seems to cut execution time by ~10%, but FFTW_ESTIMATE takes ~0.001s whereas MEASURE takes ~0.375s
    end = clock();
    printf("Time for 1st single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
    
	start = clock();
    for (i=1;i<nThreads;i++){
        allplans[i] = fftw_plan_dft_1d(fftlen, &fin[fftlen*i], &fout[fftlen*i], FFTW_BACKWARD, FFTW_MEASURE); // make the other plans, not executing them yet
    }
    end = clock();
    printf("Time for 2nd-n'th single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
	// ================================================================
	// ======== ATTACH VARS TO ARGS =======================
    ThreadArgs[0] = (void*)y;
    ThreadArgs[1] = (void*)y_i;
    ThreadArgs[2] = (void*)f_tap;
    ThreadArgs[3] = (void*)&fftlen;
    ThreadArgs[4] = (void*)&Dec;
    ThreadArgs[5] = (void*)&L;
    ThreadArgs[6] = (void*)&nprimePts;
	// fftw related
	ThreadArgs[7] = (void*)fin;
	ThreadArgs[8] = (void*)fout;
	// outputs
	ThreadArgs[9] = (void*)out_r;
	ThreadArgs[10] = (void*)out_i;
	
	
    // =============/* call the computational routine */==============
    //start threads
    for(t=0;t<nThreads;t++){
        while (t>0 && WaitForThread[t-1]==1){sleep=sleep+1; Sleep(1); printf("Slept %i..\n",sleep);}// wait for previous threads to assign ID within function
        ThreadArgs[11] = (void*)&ThreadIDList[t]; // assign the threadID
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
    
//     testing
//     for (i=fftlen*nprimePts-10;i<fftlen*nprimePts-1;i++){printf("%f %f \n", out_r[i],out_i[i]);}
//     printf("%i \n",fftlen*nprimePts);
}