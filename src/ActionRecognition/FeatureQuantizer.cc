/*
 * FeatureQuantizer.cc
 *
 *  Created on: May 5, 2017
 *      Author: richard
 */

#include "FeatureQuantizer.hh"

using namespace ActionRecognition;

const Core::ParameterEnum FeatureQuantizer::paramFeatureQuantization_("feature-quantization", "bag-of-words, fisher-vector", "bag-of-words", "feature-quantizer");

const Core::ParameterInt FeatureQuantizer::paramClusteringTrainingSamples_("training-samples", 100000, "feature-quantizer");

FeatureQuantizer::FeatureQuantizer() :
		featureQuantization_((FeatureQuantization)Core::Configuration::config(paramFeatureQuantization_)),
		nClusteringTrainingSamples_(Core::Configuration::config(paramClusteringTrainingSamples_))
{}

void FeatureQuantizer::loadModel() {
	if (featureQuantization_ == bagOfWords)
		kMeans_.loadModel();
	/* Fisher vector feature quantization */
	else // featureQuantization_ == fisherVector
		Core::Error::msg("Fisher vector feature quantization not yet implemented.") << Core::Error::abort;
}

void FeatureQuantizer::train(Features::SequenceFeatureReader& reader) {

	/* uniformly sample nClusteringTrainingSamples_ feature vectors */
	u32 i = 0;
	reader.newEpoch();
	Math::Matrix<Float> data(reader.featureDimension(), nClusteringTrainingSamples_);
	// put sampled feature vectors from reader into data matrix
	while (reader.hasSequences()) {
		const Math::Matrix<Float>& sequence = reader.next();
		for (u32 t = 0; t < sequence.nColumns(); t++) {
			if ((i % (reader.totalNumberOfFeatures() / nClusteringTrainingSamples_) == 0) && (i < nClusteringTrainingSamples_)) {
				u32 col = i / (reader.totalNumberOfFeatures() / nClusteringTrainingSamples_);
				data.copyBlockFromMatrix(sequence, 0, t, 0, col, sequence.nRows(), 1);
			}
			i++;
		}
	}
	reader.newEpoch();

	/* kMeans feature quantization */
	if (featureQuantization_ == bagOfWords) {
		kMeans_.train(data);
	}
	/* Fisher vector feature quantization */
	else { // featureQuantization_ == fisherVector
		Core::Error::msg("Fisher vector feature quantization not yet implemented.") << Core::Error::abort;
	}
}

void FeatureQuantizer::bagOfWordsQuantization(const Math::Matrix<Float>& in, Math::Vector<float>& out) {
	out.resize(kMeans_.nClusters());
	out.setToZero();
	// cluster all vectors from in, store normalized histogram in out
	Math::Vector<u32> clusterIndices;
	kMeans_.cluster(in, clusterIndices);
	for (u32 i = 0; i < clusterIndices.size(); i++)
		out.at( clusterIndices.at(i) )++;
	out.scale(1.0 / out.l1norm());
}

void FeatureQuantizer::quantize(const Math::Matrix<Float>& in, Math::Vector<float>& out) {
	/* kMeans feature quantization */
	if (featureQuantization_ == bagOfWords) {
		bagOfWordsQuantization(in, out);
	}
	/* Fisher vector feature quantization */
	else { // featureQuantization_ == fisherVector
		Core::Error::msg("Fisher vector feature quantization not yet implemented.") << Core::Error::abort;
	}
}

u32 FeatureQuantizer::outputDimension() const {
	if (featureQuantization_ == bagOfWords) {
		return kMeans_.nClusters();
	}
	else { // featureQuantization_ == fisherVector
		Core::Error::msg("Fisher vector feature quantization not yet implemented.") << Core::Error::abort;
		return 0;
	}
}
