#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <process.h>
#include <windows.h>

#include <sqlite3.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdio>

#include <tchar.h>
#include <Shlwapi.h>

#include "sqlite3.h"


#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

int insertTextFilesIntoDatabase(std::string innerdirpath, char *tablename, sqlite3 *db){
	// string declarations
	std::string innerdir = innerdirpath + "\\*";
	HANDLE ihFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA iffd;
	
	// read . and .. again
	ihFind = FindFirstFileA(innerdir.c_str(), &iffd);
	FindNextFileA(ihFind, &iffd); // read . and ..
	

	// sqlite3 declarations
	char *zErrMsg = 0;
	int rc;
	sqlite3_stmt *stmt = NULL;
	std::stringstream stmt_sstream;
	std::stringstream table_stmt_sstream;
	
	// initialize table statement
	table_stmt_sstream << "CREATE TABLE IF NOT EXISTS \"" << tablename << "\"(filename TEXT, contents TEXT);";
	printf("Initialize statement = %s\n",table_stmt_sstream.str().c_str());
	
	// initialize table
	rc = sqlite3_exec(db, table_stmt_sstream.str().c_str(), NULL, NULL, &zErrMsg);
	if (rc!=SQLITE_OK){
		printf("Failed to initialize table: %s\n", sqlite3_errmsg(db));
		return -2;
	}
	
	// initialize the insert statement
	stmt_sstream << "INSERT INTO \"" << tablename << "\" VALUES (?,?);";
	printf("Statement = %s \n", stmt_sstream.str().c_str());

	// contents variables
	std::string filepath;
	std::FILE *fp;
	std::string contents;
	
	rc = sqlite3_prepare_v2(db,
		stmt_sstream.str().c_str(),
		-1, &stmt, NULL);


	if (rc != SQLITE_OK) { printf("prepare failed: %s\n", sqlite3_errmsg(db)); }
	else {
		// start_t = GetCounter();
		sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, &zErrMsg);

		// loop over the directory
		while(FindNextFileA(ihFind, &iffd)!=0){
			// open the file and read the contents
			filepath = innerdirpath + "\\" + iffd.cFileName;
			fp = std::fopen(filepath.c_str(),"r");
			if (fp != NULL){
				contents.clear();
				std::fseek(fp,0,SEEK_END);
				contents.resize(std::ftell(fp));
				std::rewind(fp);
				std::fread(&contents[0], 1, contents.size(), fp);
				
				// bind values
				sqlite3_bind_text(stmt, 1, iffd.cFileName, -1, SQLITE_STATIC);
				sqlite3_bind_text(stmt, 2, contents.c_str(), -1, SQLITE_STATIC);
				
				// perform insert
				rc = sqlite3_step(stmt);

				sqlite3_clear_bindings(stmt); // this is to remove the values from the statement

				sqlite3_reset(stmt); // this is to make it so you can step the statement again
				
				std::fclose(fp);
			}
		}

	}
	sqlite3_exec(db, "END TRANSACTION", NULL, NULL, &zErrMsg);

	sqlite3_finalize(stmt); // THIS FREES THE MEMORY! AND THE POINTER, SO FREE-ING AT THE BOTTOM WILL CRASH
							// end_t = GetCounter();
							// printf("Time for sqlfinalize = %g ms \n", (end_t - start_t)/PCFreq);

	printf("successfully finalized statement\n");
	
	FindClose(ihFind);
	
	return 0;
}

int main(int argc, char *argv[]){
	// parameters
	std::string dirpath;
	std::string dbpath;
	if (argc != 3){
		std::cout<<"Args are (dirpath) (dbpath)"<<std::endl;
		return -1;
	}
	else{
		dirpath = argv[1];
		dbpath = argv[2];
	
		std::cout<<"Working on "<<dirpath<<std::endl;
		std::cout<<"Saving to "<<dbpath<<std::endl;
	}
	
	// initialize sqlite3 db
	sqlite3 *db;
	if (sqlite3_open_v2(dbpath.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL) != SQLITE_OK) { 
		printf("===SQLITE FAILED to open db: %s\n", sqlite3_errmsg(db)); 
		return -2;
	}
	
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	
	// iterate over the folders
	std::string dir = dirpath + "\\*";
	hFind = FindFirstFileA(dir.c_str(), &ffd);
	FindNextFileA(hFind, &ffd); // read . and ..
	std::string innerdirpath;
	
	// inner dir variables
	char tablename_cstr[MAX_PATH];
	
	
	// file variables
	std::string filepath;
	
	while(FindNextFileA(hFind,&ffd) != 0){
		innerdirpath = dirpath + "\\" + ffd.cFileName;
		std::cout<<"Now working on "<<innerdirpath<<std::endl;
		snprintf(tablename_cstr,MAX_PATH,"%s",innerdirpath.c_str());
		PathStripPathA(tablename_cstr);
		printf("Tablename will be %s\n", tablename_cstr);
		
		// run the function
		insertTextFilesIntoDatabase(innerdirpath, tablename_cstr, db);
	}

	// close the dir
	FindClose(hFind);
	
	// close the database
	sqlite3_close(db);
	
	return 0;
}