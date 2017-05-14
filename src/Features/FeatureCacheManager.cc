/*
 * FeatureCacheManager.cc
 *
 *  Created on: Apr 14, 2014
 *      Author: richard
 */

#include "FeatureCacheManager.hh"
#include <iostream>

using namespace Features;

/*
 * FeatureCachePrinter
 */
void FeatureCachePrinter::work() {
	FeatureReader reader;
	SequenceFeatureReader seqReader;
	FeatureCache::FeatureType type = FeatureCache::featureType(reader.getCacheFilename());

	switch (type) {

	case FeatureCache::vectors:
	case FeatureCache::images:
		reader.initialize();
		std::cout << std::endl;
		while (reader.hasFeatures()) {
			const Math::Vector<Float>& f = reader.next();
			std::cout << f.toString(true) << std::endl;
		}
		break;

	case FeatureCache::labels:
		reader.initialize();
		std::cout << std::endl;
		while (reader.hasFeatures()) {
			const Math::Vector<Float>& f = reader.next();
			std::cout << f.argAbsMax() << std::endl;
		}
		break;

	case FeatureCache::sequences:
	case FeatureCache::videos:
		seqReader.initialize();
		std::cout << std::endl;
		while (seqReader.hasSequences()) {
			const Math::Matrix<Float>& f = seqReader.next();
			std::cout << f.toString(true) << std::endl;
			std::cout << "#" << std::endl;
		}
		break;

	case FeatureCache::sequencelabels:
		seqReader.initialize();
		std::cout << std::endl;
		while (seqReader.hasSequences()) {
			const Math::Matrix<Float>& f = seqReader.next();
			Math::Vector<u32> labels(f.nColumns());
			f.argMax(labels);
			std::cout << labels.toString() << std::endl;
			std::cout << "#" << std::endl;
		}
		break;
	default:
		break; // this can not happen
	}
}
