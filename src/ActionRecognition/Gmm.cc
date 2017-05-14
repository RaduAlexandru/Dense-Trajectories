/*
 * Gmm.cc
 *
 *  Created on: May 12, 2017
 *      Author: ahsan
 */

#include "Gmm.hh"
#include <math.h>
#include "Math/Random.hh"
using namespace ActionRecognition;


const Core::ParameterInt Gmm::paramNumberOfGaussians_("number-of-gaussians", 64, "gmm");
const Core::ParameterInt Gmm::paramNumberOfIterations_("number-of-iterations", 100, "gmm");
const Core::ParameterString Gmm::paramWeightsFile_("model-weights-file", "", "gmm");
const Core::ParameterString Gmm::paramMeanFile_("model-mean-file", "", "gmm");
const Core::ParameterString Gmm::paramSigmaFile_("model-sigma-file", "", "gmm");

Gmm::Gmm() :
		nGaussians_(Core::Configuration::config(paramNumberOfGaussians_)),
		nIterations_(Core::Configuration::config(paramNumberOfIterations_)),
		modelWeightsFile_(Core::Configuration::config(paramWeightsFile_)),
		modelMeanFile_(Core::Configuration::config(paramMeanFile_)),
		modelSigmaFile_(Core::Configuration::config(paramSigmaFile_)) {

}

void Gmm::initializeMeans(const Math::Matrix<Float>& trainingData) {
	means_.resize(trainingData.nRows(), nGaussians_);
	std::vector<u32> indices(trainingData.nColumns());
	for (u32 i=0; i<trainingData.nColumns(); i++) {
		indices.at(i) = i;
	}
	Math::Random::initializeSRand();
	std::random_shuffle(indices.begin(), indices.end(), Math::Random::randomIntBelow);
	indices.resize(nGaussians_);
	std::sort(indices.begin(), indices.end());
	std::reverse(indices.begin(), indices.end());
	u32 c = 0;
	for (u32 i = 0; i < trainingData.nColumns(); i++) {
		if (indices.back() == i) {
			indices.pop_back();
			means_.copyBlockFromMatrix(trainingData, 0, i, 0, c, trainingData.nRows(), 1);
			c++;
		}
	}
}

void Gmm::initializeVariance(const Math::Matrix<Float>& trainingData) {
	sigmas_.resize(trainingData.nRows(), nGaussians_);
	sigmas_.fill(1);
}

void Gmm::initializeWeights() {
	weights_.resize(nGaussians_);
	weights_.fill(1.0f/nGaussians_);
}

void Gmm::initialize(const Math::Matrix<Float>& trainingData) {
	initializeMeans(trainingData);
	initializeVariance(trainingData);
	initializeWeights();
	lambdas_.resize(trainingData.nColumns(), nGaussians_);
}

Float Gmm::calculateDeterminenet(const Math::Vector<Float>& sigma) {
	Float result = 1.0f;
	for (u32 i=0; i<sigma.nRows(); i++) {
		result *= sigma.at(i);
	}
	return result;
}

void Gmm::calculateProbabilitiesGivenParams(const Math::Matrix<Float>& trainingData, Math::Matrix<Float>& result) {

	for ( u32 i=0; i<nGaussians_; i++) {

		Math::Vector<Float> sigma(sigmas_.nRows());
		sigmas_.getColumn(i, sigma);
		Math::Vector<Float> mean(trainingData.nRows());
		means_.getColumn(i, mean);

		for (u32 j=0; j<trainingData.nColumns(); j++) {
			Math::Vector<Float> feature(trainingData.nRows());
			trainingData.getColumn(i, feature);

			feature.add(mean, -1.0f);
			Math::Vector<Float> temp;
			temp.copy(feature);

			feature.elementwiseMultiplication(sigma);
			result.at(j, i) =  (1.0f/(pow((2 * M_PI), trainingData.nRows() / 2) * pow(abs(calculateDeterminenet(sigma)), 0.5))) * exp(feature.dot(temp) * (-1.0f/2.0f));

		}
	}
}

void Gmm::calculateLambdas(const Math::Matrix<Float>& probabilities) {
	for (u32 i=0; i<lambdas_.nRows(); i++) {
		for (u32 j=0; j<lambdas_.nColumns(); j++) {
			lambdas_.at(i, j) = weights_.at(j) * probabilities.at(i, j);
			Float sum = 0.0f;
			for (u32 k=0; k<nGaussians_; k++) {
				sum += weights_.at(k) * probabilities.at(i, k);
			}
			lambdas_.at(i, j) /= sum;
		}
	}
}

void Gmm::expectationStep(const Math::Matrix<Float>& trainingData) {
	Math::Matrix<Float> probabilities;
	probabilities.resize(trainingData.nColumns(), nGaussians_);
	calculateProbabilitiesGivenParams(trainingData, probabilities);

	calculateLambdas(probabilities);
}


void Gmm::getSumofLambdasForEachGaussian(Math::Vector<Float> &result) {
	for (u32 i=0; i<nGaussians_; i++) {
		result.at(i) = 0.0f;
		for (u32 j=0; j<lambdas_.nRows(); j++) {
			result.at(i) += lambdas_.at(j, i);
		}
	}
}

void Gmm::maximizationStep(const Math::Matrix<Float>& trainingData) {

	Math::Vector<Float> sums(nGaussians_);
	getSumofLambdasForEachGaussian(sums);

	//re estimate weights
	for (u32 i=0; i<nGaussians_; i++) {
		weights_.at(i) = sums.at(i) / trainingData.nColumns();
	}

	//estimate new means
	Math::Matrix<Float> newMean;
	newMean.resize(means_.nRows(), means_.nColumns());
	Math::Vector<Float> sum(trainingData.nRows());
	Math::Vector<Float> temp(trainingData.nRows());
	for (u32 j=0; j<nGaussians_; j++) {
		sum.fill(0.0f);
		for (u32 i=0; i<trainingData.nColumns(); i++) {
			trainingData.getColumn(i, temp);
			sum.add(temp, lambdas_.at(i, j));
		}
		sum.scale(1.0f/sums.at(j));
		newMean.setColumn(j, sum);
	}

	//estimate new sigmas
	Math::Matrix<Float> newSigma;
	newSigma.resize(sigmas_.nRows(), sigmas_.nColumns());
	for (u32 i=0; i<nGaussians_; i++) {
		Math::Vector<Float> mean;
		means_.getColumn(i, mean);
		sum.fill(0.0f);
		for (u32 j=0; j<trainingData.nColumns(); j++) {
			Math::Vector<Float> feature;
			trainingData.getColumn(j, feature);

			feature.add(mean, -1.0f);
			feature.elementwiseMultiplication(feature);
			sum.add(feature, lambdas_.at(j, i));
		}
		sum.scale(1.0f / sums.at(i));
		newSigma.setColumn(i, sum);
	}

	means_ = newMean;
	sigmas_ = newSigma;
}

void Gmm::train(const Math::Matrix<Float>& trainingData) {

	initialize(trainingData);
	for (u32 i =0; i< nIterations_; i++) {
		expectationStep(trainingData);
		maximizationStep(trainingData);
	}

	save();
}

void Gmm::predict(const Math::Matrix<Float>& data, Math::Matrix<Float>& result) {

	lambdas_.resize(data.nColumns(), nGaussians_);

	calculateProbabilitiesGivenParams(data, result);
	calculateLambdas(result);
	result.resize(lambdas_.nRows(), lambdas_.nColumns());
	result.copy(lambdas_);
}

void Gmm::save() {
	if (modelWeightsFile_.empty()) {
		Core::Error::msg("gmm.model-weights-file must not be empty.") << Core::Error::abort;
	}

	weights_.write(modelWeightsFile_);

	if (modelMeanFile_.empty()) {
		Core::Error::msg("gmm.model-mean-file must not be empty.") << Core::Error::abort;
	}

	means_.write(modelMeanFile_);

	if (modelSigmaFile_.empty()) {
		Core::Error::msg("gmm.model-sigma-file must not be empty.") << Core::Error::abort;
	}
	sigmas_.write(modelSigmaFile_);
}

void Gmm::load() {
	if (modelWeightsFile_.empty()) {
			Core::Error::msg("gmm.model-weights-file must not be empty.") << Core::Error::abort;
	}

	weights_.read(modelWeightsFile_);

	if (modelMeanFile_.empty()) {
		Core::Error::msg("gmm.model-mean-file must not be empty.") << Core::Error::abort;
	}

	means_.read(modelWeightsFile_);

	if (modelSigmaFile_.empty()) {
		Core::Error::msg("gmm.model-sigma-file must not be empty.") << Core::Error::abort;
	}
	sigmas_.read(modelWeightsFile_);
}
