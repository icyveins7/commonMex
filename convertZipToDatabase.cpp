#include <iostream>
#include <stdlib.h>
#include <windows.h>
#include <process.h>
#include <strsafe.h>

#include <tchar.h>
#include <Shlwapi.h>
#include <pathcch.h>

#include <zip.h>
#include <sqlite3.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "Pathcch.lib")

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

#define NUM_THREADS 1
#define BUFBYTES 1048576

struct t_data{
	int thread_t_ID;
	
	char *thread_zippaths;
	int thread_numZips;
	char *thread_outerdirpath;
	
	char *thread_dbpath;
	sqlite3 *thread_db;
	
	int *thread_successArray;
};

struct t_data t_data_array[NUM_THREADS];


unsigned __stdcall threaded_unzipFolders(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	char *zippaths = inner_data->thread_zippaths;
	int numZips = inner_data->thread_numZips;
	char *outerdirpath = inner_data->thread_outerdirpath;
	int *successArray = inner_data->thread_successArray;
	
	char *dbpath = inner_data->thread_dbpath;
	sqlite3 *db = inner_data->thread_db;

	// // folder declarations
	// HANDLE hFind;
	// WIN32_FIND_DATA ffd;
	// DWORD dwAttrs;
	
	// zip declarations
	int *errorp;
	zip_int64_t numFilesInZip;
	struct zip_stat zipFileInfo;
	zip_t *zipfile;
	zip_file_t *fileInZip;
	zip_int64_t fileBytesRead;
	
	// write file declarations
	FILE *fp;
	
	// sqlite3 declarations
	char *zErrMsg = 0;
	int rc;
	sqlite3_stmt *stmt = NULL;
	char table_stmt[256];
	char insert_stmt[256];
	
	char current_dir[MAX_PATH];
	char current_zipname[MAX_PATH];
	char current_file[MAX_PATH];
	
	char *readBuf = (char*)malloc(sizeof(char)*BUFBYTES);
	
	// size_t debug;
	
	
	
	for (int i=t_ID; i<numZips; i=i+NUM_THREADS){
		// reset zip variables
		errorp = NULL;
		
		// assume success is 1, set to 0 for any possible failure..
		successArray[i] = 1;
		
		// name the table after the zipname
		snprintf(current_zipname, MAX_PATH, "%s", &zippaths[i*MAX_PATH]); // copy into current_zipname
		PathStripPathA(current_zipname); // remove the path, leaving filename
		// PathCchRemoveExtension((PWSTR)current_zipname, MAX_PATH); // remove the .zip extension
		PathRemoveExtensionA(current_zipname); // remove the .zip extension
		
		// initialize table statement
		snprintf(table_stmt, 256, "CREATE TABLE IF NOT EXISTS \"%s\"(filename TEXT, contents TEXT);", current_zipname);
		printf("Initialize statement = %s\n",table_stmt);
		// initialize table
		rc = sqlite3_exec(db, table_stmt, NULL, NULL, &zErrMsg);
		if (rc!=SQLITE_OK){
			printf("Failed to initialize table: %s\n", sqlite3_errmsg(db));
			return -2;
		}
		// initialize the insert statement
		snprintf(insert_stmt, 256, "INSERT INTO \"%s\" VALUES (?,?);", current_zipname);
		printf("Statement = %s \n", insert_stmt);
		// prepare the insert statement
		rc = sqlite3_prepare_v2(db, insert_stmt, -1, &stmt, NULL);
		
		// snprintf(current_dir, MAX_PATH, "%s\\%s\\", outerdirpath, current_zipname); // make the full path to the folder
		// CreateDirectoryA(current_dir, NULL); // actually make the folder
		// printf("THREAD %i: Unzipping %s to %s\n", t_ID, &zippaths[i*MAX_PATH], current_dir);
		
		zipfile = zip_open(&zippaths[i*MAX_PATH], ZIP_RDONLY, errorp); // open the current zip
		if (errorp == NULL){
			printf("THREAD %i: Managed to open %s! \n", t_ID, &zippaths[i*MAX_PATH]);
			
			numFilesInZip = zip_get_num_entries(zipfile, 0);
			printf("THREAD %i: Counted files in %s \n", t_ID, &zippaths[i*MAX_PATH]);
			
			sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, &zErrMsg);
			
			for (int i = 0; i<numFilesInZip; i++){
				if (zip_stat_index(zipfile, i, 0, &zipFileInfo) !=0 ){ 
					printf("THREAD %i: Failed to open file index %i in %s! \n", t_ID, i, &zippaths[i*MAX_PATH]);
					successArray[i] = 0;
				}
				else{
					fileInZip = zip_fopen_index(zipfile, i, 0);
					if (fileInZip != NULL){ // check that the file in the zip is returned..
						fileBytesRead = zip_fread(fileInZip, readBuf, (zip_uint64_t)zipFileInfo.size);
						readBuf[fileBytesRead] = '\0'; // end it there
						
						if (fileBytesRead == (zip_uint64_t)zipFileInfo.size){ // if it's properly read, write it to disk
							// bind values
							sqlite3_bind_text(stmt, 1, zipFileInfo.name, -1, SQLITE_STATIC);
							sqlite3_bind_text(stmt, 2, readBuf, -1, SQLITE_STATIC);
							
							// perform insert
							rc = sqlite3_step(stmt);

							sqlite3_clear_bindings(stmt); // this is to remove the values from the statement

							sqlite3_reset(stmt); // this is to make it so you can step the statement again
						}
						else{
							printf("THREAD %i: Failed to write %s to db!\n", t_ID, zipFileInfo.name);
							successArray[i] = 0;
						}
						zip_fclose(fileInZip);
					}
					// printf("File at index %i is %s with uncompressedSize = %ld bytes \n", i, zipFileInfo.name, (long int)zipFileInfo.size);
				}
			}
			sqlite3_exec(db, "END TRANSACTION", NULL, NULL, &zErrMsg);
			
			zip_close(zipfile);
			printf("THREAD %i: Closed %s! \n", t_ID, &zippaths[i*MAX_PATH]);
		}
		else{
			printf("THREAD %i: Failed to open zipfile %s!\n", t_ID, &zippaths[i*MAX_PATH]);
			successArray[i] = 0;
		}
		
		sqlite3_finalize(stmt); // THIS FREES THE MEMORY! AND THE POINTER, SO FREE-ING AT THE BOTTOM WILL CRASH
	}
	
	

	//freeing
	free(readBuf);
	
	_endthreadex(0);
    return 0;
}

int main(int argc, char *argv[]){
	// == INITIALIZE TIMING ==
	int start_t = StartCounter();
	int end_t;
	
	// //reserve stuff for threads
    int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
	if (argc != 3){printf("Args are (outerdirpath) (dbpath)\n"); return 1;}
	
	TCHAR *outerdirpath = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH);
	TCHAR *outerdirpath_dir = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH);
	TCHAR *dbpath = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH);
	
	snprintf(outerdirpath, MAX_PATH, "%s", argv[1]);
	snprintf(outerdirpath_dir, MAX_PATH, "%s\\*", outerdirpath);
	printf("Set outerdir to %s \n",outerdirpath);
	snprintf(dbpath, MAX_PATH, "%s", argv[2]);
	printf("Writing to %s \n", dbpath);
	
	// initialize sqlite3 db
	sqlite3 *db;
	if (sqlite3_open_v2(dbpath, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL) != SQLITE_OK) { 
		printf("===SQLITE FAILED to open db: %s\n", sqlite3_errmsg(db)); 
		return -2;
	}
	
	int numZips = 0;
	
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
		numZips++;
	}
	printf("total %i zips\n", numZips);
	FindClose(hFind);
	
	TCHAR *zippaths = (TCHAR*)malloc(sizeof(TCHAR)*MAX_PATH * numZips);
	// redo it and this time save the folderpaths
	hFind = FindFirstFileA(outerdirpath_dir,&ffd);
	FindNextFileA(hFind, &ffd); // read the first two . and .. folders returned
	int i = 0;
	while(FindNextFileA(hFind, &ffd)!=0){
		snprintf(&zippaths[i*MAX_PATH], MAX_PATH, "%s\\%s", outerdirpath, ffd.cFileName);
		printf("zippath is %s\n", &zippaths[i*MAX_PATH]);
		i++;
	}
	
	int *successArray = (int*)malloc(sizeof(int)*numZips);
    
	

    // =============/* call the computational routine */==============
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_zippaths = zippaths;
		t_data_array[t].thread_numZips = numZips;

		t_data_array[t].thread_outerdirpath = outerdirpath;
		t_data_array[t].thread_successArray = successArray;
		
		t_data_array[t].thread_dbpath = dbpath;
		t_data_array[t].thread_db = db;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_unzipFolders,(void*)&t_data_array[t],0,NULL);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
	
	// ============== CLEANUP =================
    // close threads
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
// //         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
	
	for(i=0;i<numZips;i++){
		if(successArray[i]){
			printf("Zip %s was successful, removing zip file..\n", &zippaths[i*MAX_PATH]);
			DeleteFileA(&zippaths[i*MAX_PATH]);
		}
		else{printf("Zip %s FAILED, leaving zip file.. \n", &zippaths[i*MAX_PATH]);}
	}


	free(zippaths);
	free(outerdirpath);
	free(outerdirpath_dir);
	free(successArray);
	free(dbpath);
	
	

	
	// stop timing
	end_t = GetCounter();
	printf("\nTotal time taken : %g ms \n", (end_t - start_t)/PCFreq);
	
	return 0;
}