#include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
#include <windows.h>
#include <process.h>

#include <tchar.h>
#include <Shlwapi.h>
// #include <pathcch.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")
// #pragma comment(lib, "Pathcch.lib")

#include <zip.h>

#define NUM_THREADS 24

struct t_data{
	int thread_t_ID;
	
	char *thread_folderpaths;
	int thread_numFolders;
	int thread_deleteFlag;
	char *thread_outerdirpath;
	
	int *thread_successArray;
};

struct t_data t_data_array[NUM_THREADS];


unsigned __stdcall threaded_zipFolder(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	char *folderpaths = inner_data->thread_folderpaths;
	int numFolders = inner_data->thread_numFolders;
	int deleteFlag = inner_data->thread_deleteFlag;
	char *outerdirpath = inner_data->thread_outerdirpath;
	int *successArray = inner_data->thread_successArray;
	
	// folder declarations
	HANDLE hFind;
	WIN32_FIND_DATA ffd;
	TCHAR sourcedirpath[MAX_PATH];
	TCHAR sourcedirpath_dir[MAX_PATH];
	
	// zip declarations
	int *errorp = NULL;
	zip_error_t *errorz = NULL;
	TCHAR filename[MAX_PATH];
	TCHAR filepath[MAX_PATH];
	zip_uint64_t zipfileIdx = 0;
	TCHAR zipfilename[MAX_PATH];
	TCHAR zipfilepath[MAX_PATH];
	zip_t *zipfile;
	zip_source_t *zipsource;
	
	// return declarations
	int zipWriteCheck, zipCloseCheck;
	
	for (int i = t_ID; i<numFolders; i = i+NUM_THREADS){
		snprintf(sourcedirpath, MAX_PATH, "%s\\", &folderpaths[i*MAX_PATH]); // each sourcedirpath should end with trailing backslash
		
		snprintf(zipfilename, MAX_PATH, "%s", &folderpaths[i*MAX_PATH]); // this does not end with backslash
		// printf("thread %i, zipfilename is %s\n", t_ID, zipfilename);
		// PathCchRemoveFileSpec((PWSTR)zipfilename,MAX_PATH); // this removes the trailing backslash
		// printf("thread %i, after pathcchremove, zipfilename is %s\n", t_ID, zipfilename);
		PathStripPathA(zipfilename); // this removes the front part of the path, leaving the directory name alone
		snprintf(zipfilepath, MAX_PATH, "%s%s.zip", outerdirpath, zipfilename); // this forms the full path to the zip file
		snprintf(sourcedirpath_dir, MAX_PATH, "%s*", sourcedirpath); // this adds the * so the directory can list contents
		
		hFind = FindFirstFileA(sourcedirpath_dir,&ffd);
		FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
		
		errorp = NULL;
		errorz = NULL;
		zipfileIdx = 0;
		zipfile = zip_open(zipfilepath, ZIP_CREATE, errorp);
		zipWriteCheck = 0;
		zipCloseCheck = 0;
		
		if (errorp == NULL){
			printf("THREAD %i: Opened zip file for writing, using folder %s, zipfile at %s!\n", t_ID, sourcedirpath, zipfilepath);
			// iterate to the end of folder
			while(FindNextFileA(hFind, &ffd)!=0){
				snprintf(filename, MAX_PATH, "%s", ffd.cFileName); // get the file name
				snprintf(filepath, MAX_PATH, "%s%s", sourcedirpath, filename); // get the full path to the file
				// printf("Filepath is %s, filename is %s \n", filepath, filename);
				if((zipsource=zip_source_file_create(filepath, 0, -1, errorz)) == NULL || (zipfileIdx = zip_file_add(zipfile, filename, zipsource, ZIP_FL_OVERWRITE)) < 0){
					zip_source_free(zipsource);
					printf("Failed to add %s to zip! \n", filename);
					zipWriteCheck = 0;
				}
				else{
					zip_set_file_compression(zipfile, zipfileIdx, ZIP_CM_STORE, 1); // 1 to 9, 1 being fastest, ZIP_CM_DEFAULT, ZIP_CM_STORE
					// printf("Set file compression to fastest!\n");
					zipWriteCheck = 1;
				}
			}
		}
		
		
		if (zip_close(zipfile) < 0){
			zip_discard(zipfile);
			printf("Failed to write zip file to disk! \n");
			zipCloseCheck = 0;
		}
		else{
			zipCloseCheck = 1;
		}
		
		if (zipCloseCheck && zipWriteCheck){
			successArray[i] = 1;
			printf("THREAD %i: Completed folder %s \n", t_ID, sourcedirpath);
		}
		else{
			successArray[i] = 0;
		}

	}
	
	_endthreadex(0);
    return 0;
}



/* The gateway function */
int main( int argc, char *argv[]){

	// //reserve stuff for threads
    int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
	if (argc != 3){printf("Args are (outerdirpath) (deleteFlag)\n"); return 1;}
	
	int deleteFlag;
	sscanf (argv[2],"%d",&deleteFlag);
	
	TCHAR *outerdirpath = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH);
	TCHAR *outerdirpath_dir = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH);
	
	snprintf(outerdirpath, MAX_PATH, "%s", argv[1]);
	snprintf(outerdirpath_dir, MAX_PATH, "%s*", outerdirpath);
	printf("Set outerdir to %s \n",outerdirpath);
	
	int numFolders = 0;
	
	// folder declarations
	HANDLE hFind;
	WIN32_FIND_DATA ffd;
	char sourcedirpath[MAX_PATH];
	
	hFind = FindFirstFileA(outerdirpath_dir,&ffd);
	// printf("first file is %s \n",ffd.cFileName);
	FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
	// printf("second file is %s \n",ffd.cFileName);
	while(FindNextFileA(hFind, &ffd)!=0){
		// snprintf(sourcedirpath, MAX_PATH, "%s%s\", outerdirpath, ffd.cFileName);
		// printf("subfolderpath is %s\n", sourcedirpath);
		numFolders++;
	}
	printf("total %i folders\n", numFolders);
	FindClose(hFind);
	
	TCHAR *folderpaths = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH * numFolders);
	// redo it and this time save the folderpaths
	hFind = FindFirstFileA(outerdirpath_dir,&ffd);
	FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
	int i = 0;
	while(FindNextFileA(hFind, &ffd)!=0){
		snprintf(&folderpaths[i*MAX_PATH], MAX_PATH, "%s%s", outerdirpath, ffd.cFileName);
		printf("subfolderpath is %s\n", &folderpaths[i*MAX_PATH]);
		i++;
	}
	
	int *successArray = (int*)malloc(sizeof(int)*numFolders);
    
	

    // =============/* call the computational routine */==============
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_folderpaths = folderpaths;
		t_data_array[t].thread_numFolders = numFolders;
		t_data_array[t].thread_deleteFlag = deleteFlag;
		t_data_array[t].thread_outerdirpath = outerdirpath;
		t_data_array[t].thread_successArray = successArray;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_zipFolder,(void*)&t_data_array[t],0,NULL);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
	
	// ============== CLEANUP =================
    // close threads
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
// //         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
	
	for(i=0;i<numFolders;i++){
		if(successArray[i]){printf("Folder %s was successful\n", &folderpaths[i*MAX_PATH]);}
		else{printf("Folder %s FAILED \n", &folderpaths[i*MAX_PATH]);}
	}


	free(folderpaths);
	free(outerdirpath);
	free(outerdirpath_dir);
	free(successArray);
}