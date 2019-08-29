#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>
// #include "ipp.h"
// #include <immintrin.h> // include if trying the intrinsics code

#define NUM_THREADS 4

#define A_SQ 4.068063159076900E+13
#define B_SQ 4.040829998466145E+13
#define M_PI 3.14159265358979323846

struct t_data{
	int thread_t_ID;
	
	double *thread_latdeg;
	double *thread_londeg;
	double *thread_altm;
	int thread_numPts;

	double *thread_out;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];

unsigned __stdcall lla2ecefDeg(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	double *latdeg = inner_data->thread_latdeg;
	double *londeg = inner_data->thread_londeg;
	double *altm = inner_data->thread_altm;
	int numPts = inner_data->thread_numPts;

	double *out = inner_data->thread_out;
	// end of attached variables

	// computations
	int i;
	double lat, lon, cos_lat, cos_lon, sin_lat, sin_lon;
	double N;

	// pick point based on thread number
	for (i = t_ID; i<numPts; i=i+NUM_THREADS){
		lat = latdeg[i] * M_PI / 180.0;
		lon = londeg[i] * M_PI / 180.0;
		
		cos_lat = cos(lat);
		sin_lat = sin(lat);
		cos_lon = cos(lon);
		sin_lon = sin(lon);
		
		N = A_SQ / sqrt( A_SQ*cos_lat*cos_lat + B_SQ*sin_lat*sin_lat);
		
		out[i*3+0]  = (N + altm[i]) * cos_lat * cos_lon;
		out[i*3+1]  = (N + altm[i]) * cos_lat * sin_lon;
		out[i*3+2]  = (B_SQ*N/A_SQ + altm[i]) * sin_lat;

	}
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    // declare variables
    double *latdeg, *londeg, *altm;
	int	numPts;
	// declare outputs
	double *out;

	// //reserve stuff for threads
    int t; 
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=3){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","3 Inputs required.");
    }

    latdeg = mxGetDoubles(prhs[0]); 
	londeg = mxGetDoubles(prhs[1]); 
	altm = mxGetDoubles(prhs[2]); 
    
    numPts = (int)mxGetM(prhs[0]);

	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(3,numPts,mxREAL);
    
    /* get a pointer to the real data in the output matrix */
    out = mxGetDoubles(plhs[0]);
	
    // =============/* call the computational routine */==============
	GROUP_AFFINITY currentGroupAffinity, newGroupAffinity;
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_latdeg = latdeg;
		t_data_array[t].thread_londeg = londeg;
		t_data_array[t].thread_altm = altm;
		t_data_array[t].thread_numPts = numPts;
			
		t_data_array[t].thread_out = out;
		
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&lla2ecefDeg,(void*)&t_data_array[t],0,NULL);
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
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }

}
