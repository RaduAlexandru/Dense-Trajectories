/*
 * FeatureQuantizer.hh
 *
 *  Created on: May 5, 2017
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_FEATUREQUANTIZER_HH_
#define ACTIONRECOGNITION_FEATUREQUANTIZER_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include "Features/FeatureReader.hh"
#include "KMeans.hh"

namespace ActionRecognition {

class FeatureQuantizer
{
private:
	static const Core::ParameterEnum paramFeatureQuantization_;
	static const Core::ParameterInt paramClusteringTrainingSamples_;
	enum FeatureQuantization { bagOfWords, fisherVector };

	FeatureQuantization featureQuantization_;
	u32 nClusteringTrainingSamples_;
	KMeans kMeans_;

	void bagOfWordsQuantization(const Math::Matrix<Float>& in, Math::Vector<Float>& out);
public:
	FeatureQuantizer();
	virtual ~FeatureQuantizer() {}
	/*
	 * load the quantization model
	 */
	void loadModel();
	/*
	 * @param reader the sequence feature reader for the training sequences
	 */
	void train(Features::SequenceFeatureReader& reader);
	/*
	 * @param in the input sequence that is to be quantized (a matrix of size featureDimension x nFeatureVectors)
	 * @param out the resulting quantized feature vector
	 */
	void quantize(const Math::Matrix<Float>& in, Math::Vector<Float>& out);
	/*
	 * @return the output dimension of the feature quantization
	 */
	u32 outputDimension() const;
};

} // namespace

#endif /* ACTIONRECOGNITION_FEATUREQUANTIZER_HH_ */
