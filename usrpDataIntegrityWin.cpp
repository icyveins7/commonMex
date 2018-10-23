#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include "Shlwapi.h"

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

#define BUFSIZE MAX_PATH
#define MAX_DIR 1048576

int main(int argc, char* argv[]){
	
	TCHAR sourcedirpath[BUFSIZE];
	int sourcedirpathlen;

	TCHAR filename[BUFSIZE];
	TCHAR filepath[BUFSIZE];

	char *dirfilenames = (char*)malloc(MAX_DIR*BUFSIZE*sizeof(char));

	int fileSizeBytes;
	
	if (argc == 3){

		sourcedirpathlen = snprintf(sourcedirpath, BUFSIZE, "%s\\*", argv[1]);
		printf("Set sourcedir to %s \n",&sourcedirpath[0]);

		sscanf (argv[2],"%d",&fileSizeBytes);
		printf("Set fileSizeBytes to %i \n",fileSizeBytes);
	}
	else {
		printf("Arguments are (sourcedirpath) (fileSizeBytes) \n");
		return 1;
	}
	
	int dirIdx, i;
	// int countChar;
	// int filenamelen;
	// int filenametimes[MAX_DIR];
	long int minTime, maxTime;
	long int currTime, prevTime;

    // DIR *sourcedir;
	// struct dirent *sourceent;
	FILE *fp;
	long fileBytes;
	char singlefilename[BUFSIZE];
	char endptr[BUFSIZE];
	
	int numBreaks_badSize = 0;
	int numBreaks_missingFile = 0;
	
	HANDLE hFind;
	WIN32_FIND_DATA ffd;
	
	hFind = FindFirstFileA(sourcedirpath,&ffd);
	FindNextFileA(hFind, &ffd);
	FindNextFileA(hFind, &ffd);
	minTime = strtol(ffd.cFileName, (char**)&endptr, 10); // set both min and max times to the first one
	maxTime = strtol(ffd.cFileName, (char**)&endptr, 10);
	
	int retval = 0;
	dirIdx = 1;
	
	while(FindNextFileA(hFind, &ffd)!=0){
		currTime = strtol(ffd.cFileName, (char**)&endptr, 10);
		if (currTime<minTime){
			minTime = currTime;
		}
		if (currTime>maxTime){
			maxTime = currTime;
		}
		dirIdx++;
	}
	printf(" Earliest time = %ld, Latest time = %ld \n",minTime, maxTime);
	
	int totalExpected = maxTime - minTime + 1;
	
	for (i = minTime; i<minTime+totalExpected; i++){
		retval = 0;
		snprintf(filepath, BUFSIZE, "%s\\%ld.bin", argv[1], i);
		if (retval = PathFileExistsA(filepath)){
			fp = fopen(filepath,"rb");
			fseek(fp, 0, SEEK_END);
			fileBytes = ftell(fp);
			fclose(fp);
			if (fileBytes != fileSizeBytes){
				numBreaks_badSize++;
			}
		}
		else{
			numBreaks_missingFile++;
		}
	}
	
	printf(" Expected %i total files, got %i files, %i bad size files, %i missing files \n", totalExpected, dirIdx, numBreaks_badSize, numBreaks_missingFile);
	
	FindClose(hFind);
	
	return 0;
}