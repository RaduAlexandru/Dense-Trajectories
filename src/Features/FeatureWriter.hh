/*
 * FeatureWriter.hh
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#ifndef FEATURES_FEATUREWRITER_HH_
#define FEATURES_FEATUREWRITER_HH_

#include "Core/CommonHeaders.hh"
#include "Math/Vector.hh"
#include "FeatureCache.hh"
#include <string.h>
#include <vector>

namespace Features {

/*
 * feature writer
 */
class FeatureWriter
{
private:
	static const Core::ParameterString paramFeatureCacheFile_;
protected:
	const char* name_;

	Core::IOStream* cache_;
	std::string cacheFilename_;
	FeatureCache::FeatureType featureType_;
	u32 totalNumberOfFeatures_;
	u32 featureDim_;
	u32 nWrittenFeatures_;
	bool isInitialized_;
	bool isFinalized_;

	virtual void writeHeader();
public:
	FeatureWriter(const char* name = "features.feature-writer");
	virtual ~FeatureWriter();
	virtual void initialize(u32 totalNumberOfFeatures, u32 featureDim);
	virtual void finalize();
	virtual void write(const Math::Vector<Float>& f);
	virtual void write(const Math::Matrix<Float>& f);
};

/*
 * sequence feature writer
 */
class SequenceFeatureWriter : public FeatureWriter
{
private:
	typedef FeatureWriter Precursor;
private:
	u32 nSequences_;
	u32 nWrittenSequences_;
	virtual void writeHeader();
public:
	SequenceFeatureWriter(const char* name = "features.feature-writer");
	virtual ~SequenceFeatureWriter() {}
	virtual void initialize(u32 totalNumberOfFeatures, u32 featureDim, u32 nSequences);
	virtual void finalize();
	virtual void write(const Math::Matrix<Float>& f);
};

/*
 * label writer
 */
class LabelWriter : public FeatureWriter
{
private:
	typedef FeatureWriter Precursor;
private:
	virtual void writeHeader();
	virtual void write(const Math::Vector<Float>& f) {}
	virtual void write(const Math::Matrix<Float>& f) {}
public:
	LabelWriter(const char* name = "features.label-writer");
	virtual ~LabelWriter() {}
	virtual void initialize(u32 totalNumberOfLabels, u32 nClasses) { Precursor::initialize(totalNumberOfLabels, nClasses); }
	virtual void write(u32 label);
	virtual void write(const std::vector<u32>& labels);
};

/*
 * sequence label writer
 */
class SequenceLabelWriter : public LabelWriter
{
private:
	typedef LabelWriter Precursor;
private:
	u32 nSequences_;
	u32 nWrittenSequences_;
	virtual void writeHeader();
public:
	SequenceLabelWriter(const char* name = "features.label-writer");
	virtual ~SequenceLabelWriter() {}
	virtual void initialize(u32 totalNumberOfLabels, u32 nClasses, u32 nSequences);
	virtual void finalize();
	virtual void write(const std::vector<u32>& labels);
};

} // namespace

#endif /* FEATURES_FEATUREWRITER_HH_ */
