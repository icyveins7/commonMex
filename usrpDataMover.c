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
	TCHAR targetdirpath[BUFSIZE];
	int targetdirpathlen;
	TCHAR filename[BUFSIZE];
	TCHAR filepath[BUFSIZE];
	TCHAR targetfilepath[BUFSIZE];
	char *dirfilenames = (char*)malloc(MAX_DIR*BUFSIZE*sizeof(char));

	int storedBufferLen;
	int expectedFileSizeBytes;
	int ensureEarliestFlag;
	int copyInsteadOfMove;
	
	if (argc == 7){

		sourcedirpathlen = snprintf(sourcedirpath, BUFSIZE, "%s", argv[1]);
		printf("Set sourcedir to %s \n",&sourcedirpath[0]);

		targetdirpathlen = snprintf(targetdirpath, BUFSIZE, "%s", argv[2]);
		printf("Set targetdir to %s \n",&targetdirpath[0]);

		sscanf (argv[3],"%d",&storedBufferLen);
		printf("Set storedBufferLen to %i \n",storedBufferLen);
		
		sscanf (argv[4],"%d",&expectedFileSizeBytes);
		printf("Set expectedFileSizeBytes to %i \n",expectedFileSizeBytes);

		sscanf (argv[5], "%d", &ensureEarliestFlag);
		printf("Set ensureEarliestFlag to %i \n", ensureEarliestFlag);
		
		sscanf (argv[6], "%d", &copyInsteadOfMove);
		printf("Set copyInsteadOfMove to %i \n", copyInsteadOfMove);
	}
	else {
		printf("Arguments are (sourcedirpath) (targetdirpath) (storedBufferLen) (expectedFileSizeBytes) (ensureEarliestFlag) (copyInsteadOfMove) \n");
		return 1;
	}
	
	int dirIdx;
	int countChar;
	int filenamelen;
	int filenametimes[MAX_DIR];
	int minTime;
	int minIdx;
    DIR *sourcedir, *targetdir;
	struct dirent *sourceent;
	struct dirent *targetent;
	FILE *fp;
	long fileBytes;
	char singlefilename[BUFSIZE];
	
	BOOL failIfExists = FALSE;
	
	if (ensureEarliestFlag == 0) { // remove checks
		while (1) {
			sourcedir = opendir(sourcedirpath);
			sourceent = readdir(sourcedir);
			sourceent = readdir(sourcedir); // read twice for the . and .. folders

			if ((sourceent = readdir(sourcedir)) != NULL) {
				strcpy(singlefilename, sourceent->d_name);
				snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, singlefilename);

				if ((fp = fopen(filepath, "rb")) != NULL){
					fseek(fp, 0, SEEK_END);
					fileBytes = ftell(fp);
					fclose(fp);

					if (fileBytes == expectedFileSizeBytes) {
						printf("First file is %s \n", filepath);
						snprintf(targetfilepath, BUFSIZE, "%s%s", targetdirpath, singlefilename);
						printf("Moving file to %s \n", targetfilepath);
						if (copyInsteadOfMove){
							CopyFile(filepath, targetfilepath, failIfExists);
						}
						else{
							MoveFile(filepath, targetfilepath);
						}
					}
				}
			}
			closedir(sourcedir);

			targetdir = opendir(targetdirpath);
			targetent = readdir(targetdir);
			targetent = readdir(targetdir);

			dirIdx = 0;

			if ((targetent = readdir(targetdir)) != NULL) { // read the first file
				strcpy(singlefilename, targetent->d_name);
				snprintf(filepath, BUFSIZE, "%s%s", targetdirpath, singlefilename);

				dirIdx = 1; // you've read in 1 file

				while ((targetent = readdir(targetdir)) != NULL) { // count any other files left
					dirIdx++;
				}
			}
			closedir(targetdir);

			if (dirIdx>storedBufferLen) {
				printf("Deleting earliest file is %s \n", filepath);
				DeleteFileA(filepath);
			}

			Sleep(500);
		}

	}

	else {

		while (1) {
			sourcedir = opendir(sourcedirpath);

			dirIdx = 0;
			minTime = -1;
			while ((sourceent = readdir(sourcedir)) != NULL) {
				strcpy(&dirfilenames[dirIdx*BUFSIZE], sourceent->d_name);
				countChar = snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, &dirfilenames[dirIdx*BUFSIZE]);
				filenamelen = countChar - sourcedirpathlen;
				// printf("%s, filename length = %i \n", filepath, filenamelen);


				if (filenamelen > 10 && (fp = fopen(filepath, "rb")) != NULL) {
					fseek(fp, 0, SEEK_END);
					fileBytes = ftell(fp);
					fclose(fp);

					if (fileBytes == expectedFileSizeBytes) { //(fileBytes == 200003584){
						sscanf(&dirfilenames[dirIdx*BUFSIZE], "%d.bin", &filenametimes[dirIdx]);
						//printf("Name of file = %s, Time of file = %i, size = %i bytes, dirIdx = %i \n", &dirfilenames[dirIdx*BUFSIZE], filenametimes[dirIdx], fileBytes, dirIdx);


						if (minTime == -1) { // initialized
							minTime = filenametimes[dirIdx];
							minIdx = dirIdx;
						}
						else {
							if (filenametimes[dirIdx] < minTime) {
								minTime = filenametimes[dirIdx];
								minIdx = dirIdx;
							}
						}

						// printf("Last four chars of file is %s \n", &filepath[countChar-4]);

					}
				}
				else { printf("FAILED TO OPEN FILE %s \n", filepath); }
				dirIdx++;
			}
			closedir(sourcedir);
			printf("Total %i files in source dir \n", dirIdx);

			if (minTime != -1) {

				snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, &dirfilenames[minIdx*BUFSIZE]);
				printf("Earliest file is %s \n", filepath);
				snprintf(targetfilepath, BUFSIZE, "%s%s", targetdirpath, &dirfilenames[minIdx*BUFSIZE]);
				
				if (copyInsteadOfMove){
					printf("Copying file to %s \n", targetfilepath);
					CopyFile(filepath, targetfilepath, failIfExists);
				}
				else{
					printf("Moving file to %s \n", targetfilepath);
					MoveFile(filepath, targetfilepath);
				}
			}
			else {
				printf("Nothing to move! \n");
			}

			// delete earliest in targetdir
			targetdir = opendir(targetdirpath);

			dirIdx = 0;
			minTime = -1;
			while ((targetent = readdir(targetdir)) != NULL) {
				strcpy(&dirfilenames[dirIdx*BUFSIZE], targetent->d_name);
				countChar = snprintf(filepath, BUFSIZE, "%s%s", targetdirpath, &dirfilenames[dirIdx*BUFSIZE]);
				filenamelen = countChar - targetdirpathlen;
				// printf("%s, filename length = %i \n", filepath, filenamelen);

				if (filenamelen > 10 && (fp = fopen(filepath, "rb")) != NULL) {
					fseek(fp, 0, SEEK_END);
					fileBytes = ftell(fp);
					fclose(fp);

					if (fileBytes == expectedFileSizeBytes) { //(fileBytes == 200003584){
						sscanf(&dirfilenames[dirIdx*BUFSIZE], "%d.bin", &filenametimes[dirIdx]);
						// printf("Time of file = %i, size = %i bytes \n", filenametimes[dirIdx], fileBytes);


						if (minTime == -1) { // initialized
							minTime = filenametimes[dirIdx];
							minIdx = dirIdx;
						}
						else {
							if (filenametimes[dirIdx] < minTime) {
								minTime = filenametimes[dirIdx];
								minIdx = dirIdx;
							}
						}

						// printf("Last four chars of file is %s \n", &filepath[countChar-4]);

					}

				}
				else { printf("FAILED TO OPEN FILE %s \n", filepath); }
				dirIdx++;
			}
			closedir(targetdir);
			printf("Total %i files in targetpath \n", dirIdx);

			if (dirIdx > storedBufferLen) {
				snprintf(filepath, BUFSIZE, "%s%s", targetdirpath, &dirfilenames[minIdx*BUFSIZE]);
				printf("Deleting earliest file is %s \n", filepath);
				DeleteFileA(filepath);
			}
			else {
				printf("Nothing to delete! \n");
			}

			printf("Sleeping... \n");
			Sleep(500);
		}
	}

	return 0;
}