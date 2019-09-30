#include <iostream>
#include <stdlib.h>
#include <windows.h>
#include <process.h>
#include <strsafe.h>

#include <tchar.h>
#include <Shlwapi.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

double PCFreq = 0.0;
__int64 CounterStart = 0;

int StartCounter(){
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
    std::cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart)/1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
	return (int)CounterStart;
}

int GetCounter(){
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (int)li.QuadPart;
}

#define NUM_THREADS 3

struct t_data{
	int thread_t_ID;
	
	char *thread_folderpaths;
	int thread_numFolders;
	int thread_deleteFlag;
	char *thread_outerdirpath;
	
	int *thread_successArray;
};

struct t_data t_data_array[NUM_THREADS];


unsigned __stdcall threaded_deleteFolders(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	char *folderpaths = inner_data->thread_folderpaths;
	int numFolders = inner_data->thread_numFolders;
	// char *outerdirpath = inner_data->thread_outerdirpath;

	// folder declarations
	HANDLE hFind;
	WIN32_FIND_DATA ffd;
	// DWORD dwAttrs;
	
	char current_dir[MAX_PATH];
	char current_file[MAX_PATH];
	
	for (int i=t_ID; i<numFolders; i=i+NUM_THREADS){
		printf("THREAD %i: Working on %s \n", t_ID, &folderpaths[i*MAX_PATH]);
		
		snprintf(current_dir,MAX_PATH,"%s\\*",&folderpaths[i*MAX_PATH]);
		hFind = FindFirstFileA(current_dir,&ffd);
		FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
		while(FindNextFileA(hFind, &ffd)!=0){
			snprintf(current_file, MAX_PATH, "%s\\%s", &folderpaths[i*MAX_PATH], ffd.cFileName); // don't put the trailing backslash or they are all directories
			// dwAttrs = GetFileAttributes(current_file);// test if its a directory
			
			// printf("THREAD %i: Attempting to delete %s \n",t_ID,current_file);
			DeleteFileA(current_file);
		}
		FindClose(hFind);
	}
	
	
	_endthreadex(0);
    return 0;
}

int main(){
	// == INITIALIZE TIMING ==
	int start_t = StartCounter();
	int end_t;
	
	// for error checking
	DWORD dw; 
	bool errorCheck;
	char lpMsgBuf[128];
	
	printf("\nThis program should clear all data from a specified path, with up to 2 layers of subdirectories,\n"
		   "e.g. if C:\\mainfolder\\ is the entered path, then it will expect files ONLY in C:\\mainfolder\\a\\b\\ but not C:\\mainfolder\\a\\b\\c\\ and not C:\\mainfolder\\a\\.\n"
		   "It will also keep the subfolders intact but not the lowest subfolders e.g. it deletes C:\\mainfolder\\a\\b\\ but leaves C:\\mainfolder\\a\\. \n");
	printf("\nPlease enter the path to the highest folder: ");
	
	char topfolder[MAX_PATH];
	char topfolder_dir[MAX_PATH];
	char checkIfFolder[MAX_PATH];
	scanf("%s", topfolder);

	// check number of subfolders
	int numSubfolders = 0;
	// folder declarations
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	DWORD dwAttrs;
	
	// first iteration over main folder
	snprintf(topfolder_dir,MAX_PATH,"%s\\*",topfolder);
	hFind = FindFirstFileA(topfolder_dir,&ffd);
	FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
	while(FindNextFileA(hFind, &ffd)!=0){
		snprintf(checkIfFolder, MAX_PATH, "%s\\%s", topfolder, ffd.cFileName); // don't put the trailing backslash or they are all directories
		dwAttrs = GetFileAttributes(checkIfFolder);// test if its a directory
		
		if (dwAttrs & FILE_ATTRIBUTE_DIRECTORY){ // bitwise check on the attribute
			printf("Subfolder is %s\n", checkIfFolder);
			numSubfolders++;
		}
		// else{
			// printf("Not a subfolder: %s \n",checkIfFolder);
		// }
	}
	printf("Total %i folders\n", numSubfolders);
	errorCheck = FindClose(hFind);
	if (errorCheck == 0){
		dw = GetLastError();
		FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128, NULL);
		printf("Failed with error: %s\n", lpMsgBuf);
		return 1;
	}
	
	printf("\nCleanse all subfolders listed? (1: yes, 0: no) : ");
	int cleanseAll = 0;
	int cIdx = 0; // cleanseIdx
	scanf("%d",&cleanseAll);
	int checkCleanse;
	
	
	char *foldersToDelete = (char*)malloc(sizeof(char)*MAX_PATH*numSubfolders);
	
	// do the initial scan of the top folder to allow choices of what to delete
	
	
	hFind = FindFirstFileA(topfolder_dir,&ffd);
	FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
	if (cleanseAll == 1){ // we delete all..
		while(FindNextFileA(hFind, &ffd)!=0){
			snprintf(checkIfFolder, MAX_PATH, "%s\\%s", topfolder, ffd.cFileName); // don't put the trailing backslash or they are all directories
			dwAttrs = GetFileAttributes(checkIfFolder);// test if its a directory
			
			if (dwAttrs & FILE_ATTRIBUTE_DIRECTORY){ // bitwise check on the attribute
				snprintf(&foldersToDelete[cIdx*MAX_PATH], MAX_PATH,"%s",checkIfFolder);
				printf("%s CONTENTS MARKED FOR DELETION\n", &foldersToDelete[cIdx*MAX_PATH]);
				cIdx++;
			}
		}	
		printf("\nTotal %d folders marked for deletion\n", cIdx);

	}
	else{ // otherwise make a choice for each subfolder..
		while(FindNextFileA(hFind, &ffd)!=0){
			snprintf(checkIfFolder, MAX_PATH, "%s\\%s", topfolder, ffd.cFileName); // don't put the trailing backslash or they are all directories
			dwAttrs = GetFileAttributes(checkIfFolder);// test if its a directory
			
			if (dwAttrs & FILE_ATTRIBUTE_DIRECTORY){ // bitwise check on the attribute
				printf("Delete %s ? (1: Yes, 0: No) : ", checkIfFolder);
				scanf("%d",&checkCleanse);
				if(checkCleanse == 1){
					snprintf(&foldersToDelete[cIdx*MAX_PATH], MAX_PATH,"%s",checkIfFolder);
					printf("%s CONTENTS MARKED FOR DELETION\n", &foldersToDelete[cIdx*MAX_PATH]);
					cIdx++;
				}
			}
		}

		printf("\nTotal %d folders marked for deletion\n\n", cIdx);
	}

	errorCheck = FindClose(hFind);
	if (errorCheck == 0){
		dw = GetLastError();
		FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128, NULL);
		printf("Failed with error: %s\n", lpMsgBuf);
		return 1;
	}
	
	
	
	// now start threads within each folder (because the threading is more important within each folder for our purposes) for what has been marked..
	int t, k, s;
	HANDLE ThreadList[NUM_THREADS];
	int sIdx, sIdx_c;
	char subfolder_dir[MAX_PATH];
	char *subfoldersToDelete;
	
	// start timing
	start_t = GetCounter();
	
	// first we find the number of subfolders within each folder
	for(k=0;k<cIdx;k++){
		sIdx = 0; // reset subsubfolder count
		snprintf(subfolder_dir,MAX_PATH,"%s\\*",&foldersToDelete[k*MAX_PATH]);
		hFind = FindFirstFileA(subfolder_dir,&ffd);
		FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
		while(FindNextFileA(hFind, &ffd)!=0){
			snprintf(checkIfFolder, MAX_PATH, "%s\\%s", &foldersToDelete[k*MAX_PATH], ffd.cFileName); // don't put the trailing backslash or they are all directories
			dwAttrs = GetFileAttributes(checkIfFolder);// test if its a directory
			
			if (dwAttrs & FILE_ATTRIBUTE_DIRECTORY){ // bitwise check on the attribute
				// printf("Found subfolder %s \n", checkIfFolder);
				sIdx++;
			}
			else{
				// printf("NOT subfolder %s \n", checkIfFolder);
			}
		}
		FindClose(hFind);
		printf("Found %i subfolders in %s \n", sIdx, &foldersToDelete[k*MAX_PATH]);
		subfoldersToDelete = (char*)malloc(sizeof(char)*MAX_PATH*sIdx); // allocate for all subsubfolders
		
		if (sIdx > 0){ // only start threads on things with subfolders

			// iterate over folder again to save into long array
			sIdx_c = 0;
			hFind = FindFirstFileA(subfolder_dir,&ffd);
			FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
			while(FindNextFileA(hFind, &ffd)!=0){
				snprintf(checkIfFolder, MAX_PATH, "%s\\%s", &foldersToDelete[k*MAX_PATH], ffd.cFileName); // don't put the trailing backslash or they are all directories
				dwAttrs = GetFileAttributes(checkIfFolder);// test if its a directory
				
				if (dwAttrs & FILE_ATTRIBUTE_DIRECTORY){ // bitwise check on the attribute
					snprintf(&subfoldersToDelete[sIdx_c*MAX_PATH], MAX_PATH, "%s", checkIfFolder);
					printf("%s SUBFOLDER MARKED FOR DELETION \n", &subfoldersToDelete[sIdx_c*MAX_PATH]);
					sIdx_c++;
				}
			}
			FindClose(hFind);
			if(sIdx_c == sIdx){
				printf("Validated number of subfolders for deletion in %s : %i\n", &foldersToDelete[k*MAX_PATH], sIdx_c);

			
				// start the threads on the subfoldersToDelete
				for(t=0;t<NUM_THREADS;t++){
					t_data_array[t].thread_t_ID = t;
					
					t_data_array[t].thread_folderpaths = subfoldersToDelete;
					t_data_array[t].thread_numFolders = sIdx;
					// t_data_array[t].thread_outerdirpath = topfolder;

					ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_deleteFolders,(void*)&t_data_array[t],0,NULL);
				}
				
				WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
				
				// ============== CLEANUP =================
				// close threads
				for(t=0;t<NUM_THREADS;t++){
					CloseHandle(ThreadList[t]);
				}
			}
			
			// once you are done clearing the subfolder contents, delete the subfolder itself!
			for (s=0; s<sIdx_c; s++){
				printf("Removing subfolder %s \n",&subfoldersToDelete[s*MAX_PATH]);
				errorCheck = RemoveDirectoryA(&subfoldersToDelete[s*MAX_PATH]);
				if (errorCheck == 0){
					dw = GetLastError();
					FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128, NULL);
					printf("Failed with error: %s\n", lpMsgBuf);
					// return 1;
				}
			}
			
		}
		
		
		// freeing
		free(subfoldersToDelete);
	}
	
	// stop timing
	end_t = GetCounter();
	printf("\nTotal time taken : %g ms \n", (end_t - start_t)/PCFreq);
	
	// freeing
	free(foldersToDelete);
	
	
	return 0;
}