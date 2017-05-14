/*
 * Example.hh
 *
 *  Created on: Apr 25, 2017
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_EXAMPLE_HH_
#define ACTIONRECOGNITION_EXAMPLE_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"

namespace ActionRecognition {

class Example
{
private:
	static const Core::ParameterString paramSomeFile_;
	static const Core::ParameterInt paramSomeInt_;
	static const Core::ParameterFloat paramSomeFloat_;
	std::string filename_;
	u32 someInt_;
	Float someFloat_;

	void logMessages();
	void matrixUsage();
	void fileIO();
public:
	Example();
	virtual ~Example() {} // empty destructor
	void run();
};

} // namespace

#endif /* ACTIONRECOGNITION_EXAMPLE_HH_ */
