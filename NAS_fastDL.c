#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include <windows.h>
#include <process.h>

#include "Shlwapi.h"

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

#include <tchar.h>
#define BUFSIZE MAX_PATH
#define MAX_DIR 1048576

struct t_data{
	int thread_t_ID;
	int thread_NUM_THREADS;
	
	char *thread_sourcedirpath;
	char *thread_targetdirpath;
	long int thread_minTime;
	long int thread_maxTime;
	long int thread_totalExpected;
};

unsigned __stdcall NAS_fastDL_thread(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	int NUM_THREADS = inner_data->thread_NUM_THREADS;
	
	char *sourcedirpath = inner_data->thread_sourcedirpath;
	char *targetdirpath = inner_data->thread_targetdirpath;
	long int minTime = inner_data->thread_minTime;
	long int maxTime = inner_data->thread_maxTime;
	long int totalExpected = inner_data->thread_totalExpected;
	// =====
	TCHAR filepath[BUFSIZE];
	TCHAR targetfilepath[BUFSIZE];
	
	int retval = 0;
	int i;
	BOOL failIfExists = FALSE;
	
	for (i = minTime + t_ID; i<minTime+totalExpected; i=i+NUM_THREADS){
		retval = 0;
		snprintf(filepath, BUFSIZE, "%s%ld.bin", sourcedirpath, i); // make the sourcepath
		snprintf(targetfilepath, BUFSIZE, "%s%ld.bin", targetdirpath, i); // make the targetpath
		
		// printf("Copying %s to %s \n", filepath, targetfilepath);
		
		if (retval = PathFileExistsA(filepath)){ // if exists then copy
			CopyFile(filepath, targetfilepath, failIfExists);
		}
	}
	
	_endthreadex(0);
	return 0;
}

int main(int argc, char* argv[]){
	int t;
	int NUM_THREADS;
	
	TCHAR sourcedirpath_win[BUFSIZE];
	TCHAR sourcedirpath[BUFSIZE];
	int sourcedirpathlen;
	TCHAR targetdirpath[BUFSIZE];
	int targetdirpathlen;
	// TCHAR filename[BUFSIZE];
	char endptr[BUFSIZE];

	// char *dirfilenames = (char*)malloc(MAX_DIR*BUFSIZE*sizeof(char));
	
	if (argc == 4){
		sourcedirpathlen = snprintf(sourcedirpath, BUFSIZE, "%s", argv[1]);
		printf("Set sourcedir to %s \n",&sourcedirpath[0]);

		targetdirpathlen = snprintf(targetdirpath, BUFSIZE, "%s", argv[2]);
		printf("Set targetdir to %s \n",&targetdirpath[0]);

		sscanf (argv[3],"%d",&NUM_THREADS);
		printf("Set NUM_THREADS to %i \n",NUM_THREADS);
	}
	else {
		printf("Arguments are (sourcedirpath) (targetdirpath) (NUM_THREADS) \n");
		return 1;
	}
	
	HANDLE hFind;
	WIN32_FIND_DATA ffd;
	
	snprintf(sourcedirpath_win, BUFSIZE, "%s\\*", sourcedirpath);
	
	hFind = FindFirstFileA(sourcedirpath_win,&ffd);
	FindNextFileA(hFind, &ffd);
	FindNextFileA(hFind, &ffd);
	long int minTime = strtol(ffd.cFileName, (char**)&endptr, 10); // set both min and max times to the first one
	long int maxTime = strtol(ffd.cFileName, (char**)&endptr, 10);
	long int currTime;
	
	while(FindNextFileA(hFind, &ffd)!=0){
		currTime = strtol(ffd.cFileName, (char**)&endptr, 10);
		if (currTime<minTime){
			minTime = currTime;
		}
		if (currTime>maxTime){
			maxTime = currTime;
		}
	}
	printf(" Earliest time = %ld, Latest time = %ld \n",minTime, maxTime);
	FindClose(hFind);
	
	long int totalExpected = maxTime - minTime + 1;
	
	// start threads
	HANDLE *ThreadList = (HANDLE*)malloc(sizeof(HANDLE)*NUM_THREADS);
	struct t_data *t_data_array = (struct t_data*)malloc(sizeof(struct t_data)*NUM_THREADS);
	
	for(t=0;t<NUM_THREADS;t++){		
        t_data_array[t].thread_t_ID = t;
		t_data_array[t].thread_NUM_THREADS = NUM_THREADS;
		
		t_data_array[t].thread_sourcedirpath = (char*)sourcedirpath;
		t_data_array[t].thread_targetdirpath = (char*)targetdirpath;
		t_data_array[t].thread_minTime = minTime;
		t_data_array[t].thread_maxTime = maxTime;
		t_data_array[t].thread_totalExpected = totalExpected;
		// =====
	
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&NAS_fastDL_thread,(void*)&t_data_array[t],0,NULL);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
	
	
	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }

	free(t_data_array);
	free(ThreadList);
	
	return 0;
}
