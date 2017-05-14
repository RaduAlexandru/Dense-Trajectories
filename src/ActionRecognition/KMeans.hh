/*
 * KMeans.hh
 *
 *  Created on: Apr 26, 2017
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_KMEANS_HH_
#define ACTIONRECOGNITION_KMEANS_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"

namespace ActionRecognition {

class KMeans
{
private:
	static const Core::ParameterInt paramNumberOfClusters_;
	static const Core::ParameterInt paramNumberOfIterations_;
	static const Core::ParameterString paramModelFile_;
	u32 nClusters_;
	u32 nIterations_;
	std::string modelFile_;
	Math::Matrix<Float> means_;
public:
	KMeans();
	virtual ~KMeans() {}
	/*
	 * @param training data: a matrix of size <feature-dimension> x <number-of-training-samples> (rows: dimension, cols: samples)
	 */
	void train(const Math::Matrix<Float>& trainingData);
	/*
	 * call this function if you want to load a matrix containing cluster means
	 */
	void loadModel();
	/*
	 * @param data: the data to be clustered as a <feature-dimension> x <number-of-data-samples> matrix
	 * @param classIndices: a vector containing the cluster index for each sample from the data matrix
	 */
	void cluster(const Math::Matrix<Float>& data, Math::Vector<u32>& clusterIndices);
	/*
	 * @return the number of clusters
	 */
	u32 nClusters() const { return nClusters_; }
};

} // namespace

#endif /* ACTIONRECOGNITION_KMEANS_HH_ */
