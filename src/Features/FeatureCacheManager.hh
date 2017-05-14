/*
 * FeatureCacheManager.hh
 *
 *  Created on: Apr 14, 2014
 *      Author: richard
 */

#ifndef FEATURES_FEATURECACHEMANAGER_HH_
#define FEATURES_FEATURECACHEMANAGER_HH_

#include "Core/CommonHeaders.hh"
#include "FeatureReader.hh"

namespace Features {

class FeatureCachePrinter
{
public:
	FeatureCachePrinter() {}
	virtual ~FeatureCachePrinter() {}
	virtual void work();
};

}

#endif /* FEATURES_FEATURECACHEMANAGER_HH_ */
