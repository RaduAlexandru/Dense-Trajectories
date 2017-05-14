/*
 * Application.cc
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include "Application.hh"
#include "Configuration.hh"
#include "Log.hh"

using namespace Core;

// handler prints backtrace to std::cerr
void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

void Application::run(int argc, const char* argv[]) {

	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " <configuration>" << std::endl;
		std::cout << "<configuration> can be a configuration file, a list of parameters or both, e.g." << std::endl;
		std::cout << argv[0] << " --config=<config-file> --*.param1=value1 --parameter-prefix.param2=value2" << std::endl;
		exit(1);
	}

	// install the handler
	signal(SIGSEGV, handler);

	Core::Configuration::initialize(argc, argv);

	// run main function from derived class
	main();

	Log::finalize();
}
