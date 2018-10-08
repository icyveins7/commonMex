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

	int storedBufferLen;
	
	if (argc == 3){

		sourcedirpathlen = snprintf(sourcedirpath, BUFSIZE, "%s", argv[1]);
		printf("Set sourcedir to %s \n",&sourcedirpath[0]);

		sscanf (argv[2],"%d",&storedBufferLen);
		printf("Set storedBufferLen to %i \n",storedBufferLen);
	}
	else {
		printf("Arguments are (sourcedirpath) (storedBufferLen) \n");
		return 1;
	}
	
	int dirIdx, i;
	int countChar;
	int filenamelen;
	int filenametimes[MAX_DIR];
	int minTime;
	int minIdx;
    DIR *sourcedir;
	struct dirent *sourceent;
	FILE *fp;
	long fileBytes;
	char singlefilename[BUFSIZE];

	while (1) {
		sourcedir = opendir(sourcedirpath);
		sourceent = readdir(sourcedir);
		sourceent = readdir(sourcedir); // read twice for the . and .. folders

		dirIdx = 0;
		
		while((sourceent = readdir(sourcedir)) != NULL) {
			dirIdx++; // simply increment for every file
		}
		closedir(sourcedir);
		
		sourcedir = opendir(sourcedirpath);
		sourceent = readdir(sourcedir);
		sourceent = readdir(sourcedir); // read twice for the . and .. folders
		for (i = 0; i<dirIdx - storedBufferLen; i++){ // we restart from the beginning of folder i.e. earliest files
			if((sourceent = readdir(sourcedir)) != NULL){
				strcpy(singlefilename, sourceent->d_name);
				snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, singlefilename);
				printf("Deleting %s \n", filepath);
				DeleteFileA(filepath);
			}
		}
		closedir(sourcedir);

		Sleep(500);
	}



	return 0;
}