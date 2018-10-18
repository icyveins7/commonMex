// this should be used to check if there are breaks in a particular selection of files

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include <process.h>
#include <windows.h>
#include <tchar.h>
#define BUFSIZE MAX_PATH
#define MAX_DIR 4096

int main(int argc, char* argv[]){
	
	TCHAR sourcedirpath[BUFSIZE];
	int sourcedirpathlen;

	TCHAR filename[BUFSIZE];
	TCHAR filepath[BUFSIZE];

	char *dirfilenames = (char*)malloc(MAX_DIR*BUFSIZE*sizeof(char));

	int fileSizeBytes;
	
	if (argc == 3){

		sourcedirpathlen = snprintf(sourcedirpath, BUFSIZE, "%s", argv[1]);
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
	long int minTime;
	long int currTime, prevTime;

    DIR *sourcedir;
	struct dirent *sourceent;
	FILE *fp;
	long fileBytes;
	char singlefilename[BUFSIZE];
	char endptr[BUFSIZE];
	
	int numBreaks = 0;

	sourcedir = opendir(sourcedirpath);
	sourceent = readdir(sourcedir);
	sourceent = readdir(sourcedir); // read twice for the . and .. folders

	dirIdx = 0;
	
	while((sourceent = readdir(sourcedir)) != NULL) {
		strcpy(singlefilename, sourceent->d_name);
		snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, singlefilename);
		fp = fopen(filepath,"rb");
		fseek(fp, 0, SEEK_END);
		fileBytes = ftell(fp);
		fclose(fp);
		
		dirIdx++; // simply increment for every file
		
		if(fileBytes == fileSizeBytes){
			minTime = strtol(singlefilename, (char**)&endptr, 10);
			prevTime = minTime;
			break;
		}
	}

	// now we've found the first valid file
	// continue iterating now
	
	while ((sourceent = readdir(sourcedir)) != NULL) {
		strcpy(singlefilename, sourceent->d_name);
		snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, singlefilename);
		fp = fopen(filepath,"rb");
		fseek(fp, 0, SEEK_END);
		fileBytes = ftell(fp);
		fclose(fp);
		currTime = strtol(singlefilename, (char**)&endptr, 10);
		
		dirIdx++;
		
		// check if file size is valid and hasn't skipped
		if(fileBytes == fileSizeBytes && currTime == prevTime + 1){
			prevTime = currTime; // set the new prevTime to this file's time
		}
		else{ // otherwise we iterate until we get a valid one
			numBreaks++;
			
			while ((sourceent = readdir(sourcedir)) != NULL) {
				strcpy(singlefilename, sourceent->d_name);
				snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, singlefilename);
				fp = fopen(filepath,"rb");
				fseek(fp, 0, SEEK_END);
				fileBytes = ftell(fp);
				fclose(fp);
				currTime = strtol(singlefilename, (char**)&endptr, 10);
				
				dirIdx++;
				
				if (fileBytes == fileSizeBytes){
					printf(" Time before break = %ld, time after restored = %ld \n", prevTime, currTime);
					prevTime = currTime;
					break;
				}
			}
		}
	}
	closedir(sourcedir);

	printf(" Start time = %ld \n End time = %ld \n Number of breaks = %i \n Total files = %i \n", minTime, currTime, numBreaks, dirIdx);

	return 0;
}