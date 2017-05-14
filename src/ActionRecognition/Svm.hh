/*
 * Svm.hh
 *
 *  Created on: May 2, 2017
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_SVM_HH_
#define ACTIONRECOGNITION_SVM_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include "FeatureQuantizer.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

namespace ActionRecognition {

class Svm
{
private:
	static const Core::ParameterString paramModelDirectory_;
	static const Core::ParameterInt paramIterations_;

	std::string modelDirectory_;
	u32 iterations_;
	FeatureQuantizer featureQuantizer_;

	void trainSVM(cv::Mat& data, const Math::Vector<u32>& labels, u32 nClasses);
public:
	Svm();
	virtual ~Svm() {}
	/*
	 * train the feature quantizer and the SVM
	 */
	void train();
	/*
	 * classify the input features
	 */
	void classify();
};

} // namespace


#endif /* ACTIONRECOGNITION_SVM_HH_ */
