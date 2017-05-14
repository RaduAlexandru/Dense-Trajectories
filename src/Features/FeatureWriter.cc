/*
 * FeatureWriter.cc
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#include "FeatureWriter.hh"
#include <stdio.h>

using namespace Features;

/*
 * FeatureWriter
 */
const Core::ParameterString FeatureWriter::paramFeatureCacheFile_("feature-cache", "", "features.feature-writer");

FeatureWriter::FeatureWriter(const char* name) :
		name_(name),
		cache_(0),
		cacheFilename_(Core::Configuration::config(paramFeatureCacheFile_, name_)),
		featureType_(FeatureCache::vectors),
		totalNumberOfFeatures_(0),
		featureDim_(0),
		nWrittenFeatures_(0),
		isInitialized_(false),
		isFinalized_(false)
{}

FeatureWriter::~FeatureWriter() {
	if (!isFinalized_)
		finalize();
}

void FeatureWriter::writeHeader() {
	require(!isFinalized_);
	require(isInitialized_);
	(*cache_) << "#vectors" << Core::IOStream::endl;
	(*cache_) << totalNumberOfFeatures_ << " " << featureDim_ << Core::IOStream::endl;
}

void FeatureWriter::initialize(u32 totalNumberOfFeatures, u32 featureDim) {
	require(!isFinalized_);
	totalNumberOfFeatures_ = totalNumberOfFeatures;
	featureDim_ = featureDim;
	if (Core::Utils::isGz(cacheFilename_))
		cache_ = new Core::CompressedStream(cacheFilename_, std::ios::out);
	else
		cache_ = new Core::AsciiStream(cacheFilename_, std::ios::out);
	isInitialized_ = true;
	writeHeader();
}

void FeatureWriter::finalize() {
	require(!isFinalized_);
	require(isInitialized_);
	if (nWrittenFeatures_ != totalNumberOfFeatures_) {
		Core::Error::msg("FeatureWriter::finalize: announced to write ") << totalNumberOfFeatures_ << " vectors but "
		 << nWrittenFeatures_ << " vectors written." << Core::Error::abort;
	}
	cache_->close();
	delete cache_;
	Core::Log::openTag("features.feature-writer");
	Core::Log::os("Wrote cache ") << cacheFilename_;
	Core::Log::closeTag();
	isFinalized_ = true;
}

void FeatureWriter::write(const Math::Vector<Float>& f) {
	Math::Matrix<Float> m(f.nRows(), 1);
	m.copy(f.begin());
	write(m);
}

void FeatureWriter::write(const Math::Matrix<Float>& f) {
	require(!isFinalized_);
	require(isInitialized_);
	if (f.nRows() != featureDim_)
		Core::Error::msg("FeatureWriter::write: feature dimension mismatch (") << f.nRows() << " vs. " << featureDim_ << ")" << Core::Error::abort;
	(*cache_) << f.toString(true) << Core::IOStream::endl;
	nWrittenFeatures_ += f.nColumns();
}

/*
 * SequenceFeatureWriter
 */

SequenceFeatureWriter::SequenceFeatureWriter(const char* name) :
		Precursor(name),
		nSequences_(0),
		nWrittenSequences_(0)
{
	featureType_ = FeatureCache::sequences;
}

void SequenceFeatureWriter::writeHeader() {
	require(!isFinalized_);
	require(isInitialized_);
	(*cache_) << "#sequences" << Core::IOStream::endl;
	(*cache_) << totalNumberOfFeatures_ << " " << featureDim_ << " " << nSequences_ << Core::IOStream::endl;
}

void SequenceFeatureWriter::initialize(u32 totalNumberOfFeatures, u32 featureDim, u32 nSequences) {
	nSequences_ = nSequences;
	Precursor::initialize(totalNumberOfFeatures, featureDim);
}

void SequenceFeatureWriter::finalize() {
	if (nWrittenSequences_ != nSequences_) {
		Core::Error::msg("SequenceFeatureWriter::finalize: announced to write ") << nSequences_ << " sequences but "
		 << nWrittenSequences_ << " sequences written." << Core::Error::abort;
	}
	Precursor::finalize();
}

void SequenceFeatureWriter::write(const Math::Matrix<Float>& f) {
	Precursor::write(f);
	(*cache_) << "#" << Core::IOStream::endl;
	nWrittenSequences_++;
}

/*
 * LabelWriter
 */

LabelWriter::LabelWriter(const char* name) :
		Precursor(name)
{
	featureType_ = FeatureCache::labels;
}

void LabelWriter::writeHeader() {
	require(!isFinalized_);
	require(isInitialized_);
	(*cache_) << "#labels" << Core::IOStream::endl;
	(*cache_) << totalNumberOfFeatures_ << " " << featureDim_ << Core::IOStream::endl;
}

void LabelWriter::write(u32 label) {
	std::vector<u32> labels;
	labels.push_back(label);
	write(labels);
}

void LabelWriter::write(const std::vector<u32>& labels) {
	require(!isFinalized_);
	require(isInitialized_);
	for (u32 i = 0; i < labels.size(); i++) {
		if (labels.at(i) >= featureDim_)
			Core::Error::msg("LabelWriter::write: label is ") << labels.at(i) << " but must be smaller than " << featureDim_ << "." << Core::Error::abort;
		(*cache_) << labels.at(i) << Core::IOStream::endl;
	}
	nWrittenFeatures_ += labels.size();
}

/*
 * SequenceLabelWriter
 */

SequenceLabelWriter::SequenceLabelWriter(const char* name) :
		Precursor(name),
		nSequences_(0),
		nWrittenSequences_(0)
{
	featureType_ = FeatureCache::sequencelabels;
}

void SequenceLabelWriter::writeHeader() {
	require(!isFinalized_);
	require(isInitialized_);
	(*cache_) << "#sequencelabels" << Core::IOStream::endl;
	(*cache_) << totalNumberOfFeatures_ << " " << featureDim_ << " " << nSequences_ << Core::IOStream::endl;
}

void SequenceLabelWriter::initialize(u32 totalNumberOfLabels, u32 nClasses, u32 nSequences) {
	nSequences_ = nSequences;
	Precursor::initialize(totalNumberOfLabels, nClasses);
}

void SequenceLabelWriter::finalize() {
	if (nWrittenSequences_ != nSequences_) {
		Core::Error::msg("SequenceLabelWriter::finalize: announced to write ") << nSequences_ << " sequences but "
		 << nWrittenSequences_ << " sequences written." << Core::Error::abort;
	}
	Precursor::finalize();
}

void SequenceLabelWriter::write(const std::vector<u32>& labels) {
	Precursor::write(labels);
	(*cache_) << "#" << Core::IOStream::endl;
	nWrittenSequences_++;
}
