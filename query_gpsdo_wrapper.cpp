#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <regex>
#include <windows.h>

std::string exec(const char* cmd){
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose); // add _ for windows to popen/pclose
	if (!pipe){
		throw std::runtime_error("popen() failed!\n");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr ){
		result += buffer.data();
		std::cout<<buffer.data();
	}
	return result;
}

int execSuccess(const char* cmd){
	std::array<char, 128> buffer;
	std::string result;
	
	std::smatch m;
	std::regex e ("GPS Locked");
	
	int success = 0;
	std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose); // add _ for windows to popen/pclose
	if (!pipe){
		throw std::runtime_error("popen() failed!\n");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr ){
		result += buffer.data();
		std::cout<<buffer.data(); // print as it goes..
	}
	
	if ( std::regex_search (result,m,e)){
		success = 1;
	}
	
	return success;
}

int main(){
	std::string cmd = "\"C:\\Program Files (x86)\\National Instruments\\NI-USRP\\utilities\\query_gpsdo_sensors.exe\"";
	// std::string result = exec(cmd.c_str());
	
	int success = 0;
	
	while (!success){
		success = execSuccess(cmd.c_str());
		if (!success){
			std::cout<<"Retrying in 2 seconds..\n";
			Sleep(2000);
		}
	}
	
	std::cout<<std::endl<<"================================ GPS READY ================================"<<std::endl;
	
	
	
	return 0;
}