/*
 * Performs filtering and downsampling on data.
 *
 * mexIppDecimate(x,numTaps,Wn,Dec)
 * Inputs: signal, filter tap number, frequency cutoff, decimation factor
 *
 * New code directly implementing filter taps (fir1 equivalent), filtering, and downsampling from IPP functions.
 *
 *
*/

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"


void ippDecimate(int numTaps, double Wn, int Dec, double *x_r, double *x_i, double *out_r, double *out_i, int siglen){
    // declarations
    Ipp64f *pTaps, *pTaps_imag;
    Ipp64fc *pDly;
    Ipp64fc *x_c, *filtered, *downsampled, *pTaps_c;
    Ipp8u *gen_pBuffer, *SR_pBuffer;
    IppsFIRSpec_64fc *pSpec;
    
    int specSize, gen_bufSize, SR_bufSize;
    int downsampledLen, pPhase;
    
    // downsampling args
    downsampledLen = siglen/Dec;
    pPhase = 0;
    
    // allocations
    pTaps = (Ipp64f*)ippsMalloc_64f_L(numTaps);
	pDly = (Ipp64fc*)ippsMalloc_64fc_L((numTaps-1));
	pTaps_imag = (Ipp64f*)ippsMalloc_64f_L(numTaps);
    pTaps_c = (Ipp64fc*)ippsMalloc_64fc_L(numTaps); // make a complex taps array
    x_c = (Ipp64fc*)ippsMalloc_64fc_L(siglen);
    filtered = (Ipp64fc*)ippsMalloc_64fc_L(siglen);
    downsampled = (Ipp64fc*)ippsMalloc_64fc_L(downsampledLen);

    // start making filter taps
    ippsFIRGenGetBufferSize(numTaps, &gen_bufSize);
    gen_pBuffer = ippsMalloc_8u_L(gen_bufSize);
    ippsFIRGenLowpass_64f(Wn/2, pTaps, numTaps, ippWinHamming, ippTrue, gen_pBuffer); // generate the filter coefficients
    
    // make the filter 
    ippsFIRSRGetSize(numTaps, ipp64fc, &specSize, &SR_bufSize);
    SR_pBuffer = ippsMalloc_8u_L(SR_bufSize);
    pSpec = (IppsFIRSpec_64fc*)ippsMalloc_8u_L(specSize);
    ippsZero_64f(pTaps_imag,numTaps); // zero out the imag components
    ippsRealToCplx_64f(pTaps,pTaps_imag,pTaps_c,numTaps); // convert filter taps to complex
    ippsFIRSRInit_64fc(pTaps_c, numTaps, ippAlgFFT, pSpec); // initialize filter
    
    // do the filtering
    ippsRealToCplx_64f((Ipp64f*)x_r, (Ipp64f*)x_i, x_c, siglen); // convert original sig to interleaved
    ippsFIRSR_64fc(x_c, filtered, siglen, pSpec, NULL, pDly, SR_pBuffer);
    
    // downsample
    ippsSampleDown_64fc(filtered, siglen, downsampled, &downsampledLen, Dec, &pPhase);
    
    // convert back to matlab split arrays
    ippsCplxToReal_64fc(downsampled, (Ipp64f*)out_r, (Ipp64f*)out_i, downsampledLen);
    
    // freeing
    ippsFree(pTaps);
    ippsFree(pDly);
    ippsFree(pTaps_imag);
    ippsFree(pTaps_c);
    ippsFree(x_c);
    ippsFree(filtered);
    ippsFree(downsampled);
    ippsFree(pSpec);
    ippsFree(gen_pBuffer);
    ippsFree(SR_pBuffer);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // declare variables
    double *x_r, *x_i;
    int numTaps, siglen, Dec;
    double Wn;
	// declare outputs
	double *out_r, *out_i;
	
    
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	x_r = mxGetPr(prhs[0]); 
	x_i = mxGetPi(prhs[0]);
    numTaps = (int)mxGetScalar(prhs[1]);
    Wn = (double)mxGetScalar(prhs[2]);
    Dec = (int)mxGetScalar(prhs[3]); // decimation factor
    
    siglen = (mwSize)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
	
	if (siglen%Dec!=0){
        mexErrMsgTxt("Decimation factor is not a factor of signal length!");
    }
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix((mwSize)1,(mwSize)(siglen/Dec),mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out_r = mxGetPr(plhs[0]); // CHECK GET PR OR GET PI BASED ON WHAT YOU ALLOCATED
    out_i = mxGetPi(plhs[0]); // THIS IS PI

    // =============/* call the computational routine */==============
    ippInit();
    ippDecimate(numTaps, Wn, Dec, x_r, x_i, out_r, out_i, siglen);
}