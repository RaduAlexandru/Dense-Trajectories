/*
 * Application.hh
 *
 *  Created on: Jun 29, 2015
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_APPLICATION_HH_
#define ACTIONRECOGNITION_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"

namespace ActionRecognition {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	enum Actions { example, stip, svmTrain, svmClassify };
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace

#endif /* ACTIONRECOGNITION_APPLICATION_HH_ */
