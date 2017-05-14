/*
 * Error.cc
 *
 *  Created on: May 20, 2016
 *      Author: richard
 */

#include "Error.hh"

using namespace Core;

Error* Error::theInstance_ = 0;

Error& Error::getInstance() {
	if (theInstance_ == 0)
		theInstance_ = new Error;
	return *theInstance_;
}

void Error::abort(std::ostream& stream) {
	std::cerr << std::endl << "Abort." << std::endl;
	exit(1);
}

Error& Error::msg(const char* msg) {
	std::cerr << std::endl << "ERROR:" << std::endl << msg;
	return getInstance();
}

Error& Error::operator<<(void (*fptr)(std::ostream&)) { fptr(std::cerr); return getInstance(); }

Error& Error::operator<<(u8 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(u32 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(u64 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(s8 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(s32 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(s64 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(f32 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(f64 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(bool n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(char n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(const char* n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(const std::string& n) { std::cerr << n; return getInstance(); }
