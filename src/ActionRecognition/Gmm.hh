/*
 * Gmm.h
 *
 *  Created on: May 12, 2017
 *      Author: ahsan
 */

#ifndef ACTIONRECOGNITION_GMM_HH_
#define ACTIONRECOGNITION_GMM_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"

namespace ActionRecognition {

class Gmm
{
private:
	static const Core::ParameterInt paramNumberOfGaussians_;
	static const Core::ParameterInt paramNumberOfIterations_;
	static const Core::ParameterString paramWeightsFile_;
	static const Core::ParameterString paramMeanFile_;
	static const Core::ParameterString paramSigmaFile_;

	u32 nGaussians_;
	u32 nIterations_;
	std::string modelWeightsFile_;
	std::string modelMeanFile_;
	std::string modelSigmaFile_;

	/*
	 * a matrix of size <feature-dimention> x <number of gaussians>
	 */
	Math::Matrix<Float> means_;
	/*
	 * a matrix of size <feature-dimention> x <number of gaussians>
	 */
	Math::Matrix<Float> sigmas_;
	/*
	 * a vector of size <number of gaussians>
	 */
	Math::Vector<Float> weights_;
	/*
	 * a matrix of size <number of features> x <number of gaussians>
	 */
	Math::Matrix<Float> lambdas_;

	void initialize(const Math::Matrix<Float>& trainingData);
	void initializeMeans(const Math::Matrix<Float>& trainingData);
	void initializeVariance(const Math::Matrix<Float>& trainingData);
	void initializeWeights();

	Float calculateDeterminenet(const Math::Vector<Float> &sigma);
	void calculateProbabilitiesGivenParams(const Math::Matrix<Float> &trainingData, Math::Matrix<Float> &result);
	void calculateLambdas(const Math::Matrix<Float>& probabilities);
	void expectationStep(const Math::Matrix<Float>& trainingData);

	void getSumofLambdasForEachGaussian(Math::Vector<Float>& result);
	void maximizationStep(const Math::Matrix<Float>& trainingData);

	void calculateInverse(const Math::Vector<Float>& diagonalMat, Math::Vector<Float>& result);
public:
	Gmm();
	virtual ~Gmm() {}

	/*
	 * @param training data: a matrix of size <feature-dimension> x <number-of-training-samples> (rows: dimension, cols: samples)
	 */
	void train(const Math::Matrix<Float>& trainingData);

	/*
	 * @param data: a matrix of size <feature-dimension> x <number-of-training-samples> (rows: dimension, cols: samples)
	 * @param result: a mtrix of size <number-of-training-samples> x <number-of-gaussians>
	 */
	void predict(const Math::Matrix<Float>& data, Math::Matrix<Float>& result);

	void save();
	void load();

	/*
	 * returns: a matrix of size <feature-dimention> x <number of gaussians>
	 */
	const Math::Matrix<Float>& mean() {
		return means_;
	}

	/*
	 * returns: a matrix of size <feature-dimention> x <number of gaussians>
	 */
	const Math::Matrix<Float>& sigma() {
		return sigmas_;
	}

	/*
	 * returns: a vector of size <number of gaussians>
	 */
	const Math::Vector<Float>& weights() {
		return weights_;
	}
};

} // namespace



#endif /* ACTIONRECOGNITION_GMM_HH_ */
