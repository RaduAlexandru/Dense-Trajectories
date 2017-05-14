/*
 * Application.hh
 *
 *  Created on: Apr 14, 2014
 *      Author: richard
 */

#ifndef FEATURES_APPLICATION_HH_
#define FEATURES_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"
#include "FeatureCacheManager.hh"

namespace Features {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	enum Actions { none, printCache };
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace

#endif /* APPLICATION_HH_ */
