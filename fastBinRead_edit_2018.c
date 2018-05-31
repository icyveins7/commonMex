// remember to compile with -R2018a command!
#include "stdio.h"
#include <string.h>
#include <windows.h>
#include <process.h>
#include <limits.h>
#include "mex.h"
#include "ipp.h"

// should be enough to max out the 10gbps
#define NUM_THREADS 4 

struct thread_data{
	int thread_t_ID;
    int thread_fs;
    int thread_total_num_of_cells;
	FILE **thread_fp;
    Ipp16sc *thread_rawdata;
    mxComplexDouble *thread_data; // final output
};

struct thread_data thread_data_array[NUM_THREADS];
HANDLE ThreadList[NUM_THREADS];

unsigned __stdcall thread_reader(void *pArgs){
    struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
    
    int t_ID = inner_data->thread_t_ID;
    int fs = inner_data->thread_fs;
    int total_num_of_cells = inner_data->thread_total_num_of_cells;
    FILE **fp = inner_data->thread_fp;
    Ipp16sc *rawdata = inner_data->thread_rawdata;
    mxComplexDouble *data = inner_data->thread_data;
    
    int i;
    
    for (i = t_ID; i<total_num_of_cells; i=i+NUM_THREADS){ // each thread takes one file
        fread(&rawdata[i*fs],sizeof(Ipp16sc),fs,fp[i]);
        // go from 16s to doubles
        ippsConvert_16s64f_Sfs((Ipp16s*)&rawdata[i*fs], (Ipp64f*)&data[i*fs], fs*2, 0); // scale factor is 2^lastArg so use 2^0 = 1, convert for one file, so fs*2 total doubles since complex
    }

    _endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	const mxArray *cell_element_ptr;
	char c_array[200];
	int i;
	int total_num_of_cells, buflen;
	int status;
    FILE **fp;
    Ipp16sc *rawdata;
    
    mxComplexDouble *data;
    int fs;
    
    // thread stuff
    int t;
    
    /*Extract the cotents of MATLAB cell into the C array*/
	total_num_of_cells = (int)mxGetNumberOfElements(prhs[0]);
    fs = (int)mxGetScalar(prhs[1]);
    
    /* check for proper number of arguments */
	if (nrhs!=2){
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
	}
    	
	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,fs*total_num_of_cells,mxCOMPLEX);
    
    /* get a pointer to the real data in the output matrix */
    data = mxGetComplexDoubles(plhs[0]);

    /* the computations */
    ippInit();
    mexPrintf("This mexfile uses 4 threads to fully saturate the 10Gbps link,\n does not convert to split real and imaginary double arrays since R2018a no longer requires it.\n");
    
    // open file pointers
    fp = (FILE**)mxMalloc(sizeof(FILE*)*total_num_of_cells);
    rawdata = (Ipp16sc*)ippsMalloc_16sc_L(total_num_of_cells*fs);
	for(i=0;i<total_num_of_cells;i++){
		cell_element_ptr = mxGetCell(prhs[0],i);
		buflen = (int)(mxGetN(cell_element_ptr)*sizeof(mxChar)+1);
		status = mxGetString(cell_element_ptr,c_array,buflen);
        fp[i] = fopen(c_array,"rb");
        if (fp[i]!=NULL){
			mexPrintf("Opened file %s\n", c_array);
		}
		else{
			printf("Failed to open %s!\n", c_array);
            mexErrMsgTxt("MEX CLOSED DUE TO INACCESSIBLE FILE\n");
		}
	}
    
    // start threads
    for (t=0; t<NUM_THREADS; t++){
		thread_data_array[t].thread_t_ID = t;
		thread_data_array[t].thread_fs = fs;
		thread_data_array[t].thread_total_num_of_cells = total_num_of_cells;
        thread_data_array[t].thread_fp = fp;
        thread_data_array[t].thread_rawdata = rawdata;
        thread_data_array[t].thread_data = data;

		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&thread_reader,(void*)&thread_data_array[t],0,NULL);
		printf("Starting thread %i \n", thread_data_array[t].thread_t_ID);
	}
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE); // close threads after waiting for all to end
    for(t=0;t<NUM_THREADS+1;t++){
		CloseHandle(ThreadList[t]);
	}

    for (i=0; i<total_num_of_cells; i++){fclose(fp[i]);} // close all file pointers
    mxFree(fp);
	ippsFree(rawdata);
}
