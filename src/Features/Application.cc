/*
 * Application.cc
 *
 *  Created on: Apr 14, 2014
 *      Author: richard
 */

#include "Application.hh"
#include "FeatureCacheManager.hh"
#include <iostream>

using namespace Features;

APPLICATION(Features::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, print-cache",
		"none");

void Application::main() {
	switch (Core::Configuration::config(paramAction_)) {
	case printCache:
		{
		FeatureCachePrinter printer;
		printer.work();
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}
