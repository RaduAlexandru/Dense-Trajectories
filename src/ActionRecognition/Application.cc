/*
 * Application.cc
 *
 *  Created on: Jun 29, 2015
 *      Author: richard
 */

#include "Application.hh"
#include <iostream>
#include "Example.hh"
#include "Stip.hh"
#include "Math/Random.hh"
#include "KMeans.hh"
#include "Svm.hh"
#include "Dense.hh"

using namespace ActionRecognition;

APPLICATION(ActionRecognition::Application)

const Core::ParameterEnum Application::paramAction_("action",
        "example, stip, svm-train, svm-classify,dense", // list of possible actions
		"example"); // the default

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case example:
	{
		Example example;
		example.run();
	}
	break;
	case stip:
	{
		Stip stip;
		stip.run();
	}
	break;
	case svmTrain:
	{
		Svm svm;
		svm.train();
	}
	break;
	case svmClassify:
	{
		Svm svm;
		svm.classify();
	}
	break;
    case dense:
    {
        Dense dense;
        dense.run();
    }
    break;
	default:
		Core::Error::msg("No action given.") << Core::Error::abort;
	}
}
