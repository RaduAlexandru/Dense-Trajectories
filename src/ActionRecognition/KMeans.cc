/*
 * KMeans.cc
 *
 *  Created on: Apr 28, 2017
 *      Author: richard
 */

#include "KMeans.hh"
#include "Math/Random.hh"

using namespace ActionRecognition;

const Core::ParameterInt KMeans::paramNumberOfClusters_("number-of-clusters", 4000, "kmeans");

const Core::ParameterInt KMeans::paramNumberOfIterations_("number-of-iterations", 100, "kmeans");

const Core::ParameterString KMeans::paramModelFile_("model-file", "", "kmeans");

KMeans::KMeans() :
		nClusters_(Core::Configuration::config(paramNumberOfClusters_)),
		nIterations_(Core::Configuration::config(paramNumberOfIterations_)),
		modelFile_(Core::Configuration::config(paramModelFile_))
{
	if (modelFile_.empty())
		Core::Error::msg("kmeans.model-file must not be empty.") << Core::Error::abort;
}

void KMeans::train(const Math::Matrix<Float>& trainingData) {
	Core::Log::openTag("kmeans");

	u32 featureDim = trainingData.nRows();
	u32 nFeatures = trainingData.nColumns();
	// random initialization if no mean file has been loaded
	if (means_.size() == 0) {
		means_.resize(featureDim, nClusters_);
		// random initialization of cluster centers
		for (u32 c = 0; c < nClusters_; c++) {
			u32 n = Math::Random::randomInt((u32)0, nFeatures-1);
			means_.copyBlockFromMatrix(trainingData, 0, n, 0, c, featureDim, 1);
		}
	}

	// train
	for (u32 i = 0; i < nIterations_; i++) {
		Core::Log::os("iteration ") << i;
		// cluster
		Math::Vector<u32> clusterIndices(nFeatures);
		cluster(trainingData, clusterIndices);
		// re-estimate cluster means
		means_.setToZero();
		std::vector<u32> count(nClusters_, 0);
		for (u32 n = 0; n < nFeatures; n++)
			count.at(clusterIndices.at(n))++;
		for (u32 n = 0; n < nFeatures; n++)
			means_.addBlockFromMatrix(trainingData, 0, n, 0, clusterIndices.at(n), featureDim, 1, 1.0 / count.at(clusterIndices.at(n)));
	}

	// save model
	means_.write(modelFile_);

	Core::Log::closeTag();
}

void KMeans::loadModel() {
	means_.read(modelFile_);
	require_eq(means_.nColumns(), nClusters_);
}

void KMeans::cluster(const Math::Matrix<Float>& data, Math::Vector<u32>& clusterIndices) {
	if (means_.size() == 0)
		loadModel();
	require_eq(data.nRows(), means_.nRows());
	require_eq(means_.nColumns(), nClusters_);

	clusterIndices.resize(data.nColumns());

#pragma omp parallel for
	for (u32 n = 0; n < data.nColumns(); n++) {

		Float minDist = Types::max<Float>();
		u32 bestCluster = 0;
		for (u32 c = 0; c < nClusters_; c++) {
			Float dist = 0;
			for (u32 d = 0; d < data.nRows(); d++) {
				dist += (data.at(d, n) - means_.at(d, c)) * (data.at(d, n) - means_.at(d, c));
				if (dist > minDist) break;
			}
			if (dist < minDist) {
				minDist = dist;
				bestCluster = c;
			}
			clusterIndices.at(n) = bestCluster;
		}
	}
}
