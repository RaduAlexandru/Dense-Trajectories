/*
 * Svm.cc
 *
 *  Created on: May 2, 2017
 *      Author: richard
 */

#include "Svm.hh"
#include "Features/AlignedFeatureReader.hh"
#include <sstream>

using namespace ActionRecognition;

const Core::ParameterString Svm::paramModelDirectory_("model-directory", "", "svm");

const Core::ParameterInt Svm::paramIterations_("iterations", 1000, "svm");

Svm::Svm() :
		modelDirectory_(Core::Configuration::config(paramModelDirectory_)),
		iterations_(Core::Configuration::config(paramIterations_))
{
	if (modelDirectory_.empty())
		Core::Error::msg("model-directory not specified.") << Core::Error::abort;
}

void Svm::trainSVM(cv::Mat& data, const Math::Vector<u32>& labels, u32 nClasses) {
    // Set up SVM parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, iterations_, 1e-8);

	// train an SVM for each class (one-against-all)
	for (u32 c = 0; c < nClasses; c++) {
		// create label file with -1 (negative class) and 1 (positive class)
		Math::Vector<Float> l(labels.size());
		for (u32 i = 0; i < labels.size(); i++)
			l.at(i) = (labels.at(i) == c ? 1.0 : -1.0);
		cv::Mat labelsMat(labels.size(), 1, CV_32FC1, l.begin());
		// actual SVM training
	    CvSVM SVM;
	    SVM.train_auto(data, labelsMat, cv::Mat(), cv::Mat(), params);
	    // save model
	    std::stringstream filename;
	    filename << modelDirectory_ << "/svm-" << c;
	    SVM.save(filename.str().c_str());
	}
}

void Svm::train() {
	Features::LabeledSequenceFeatureReader reader;
	reader.initialize();

	featureQuantizer_.train(reader);

	/* train support vector machine */
	// quantize training data and put into one big matrix
	Math::Matrix<Float> data(featureQuantizer_.outputDimension(), reader.totalNumberOfSequences());
	Math::Vector<u32> labels(reader.totalNumberOfSequences());
	for (u32 i = 0; i < reader.totalNumberOfSequences(); i++) {
		Math::Vector<Float> f;
		featureQuantizer_.quantize(reader.next(), f);
		data.setColumn(i, f);
		labels.at(i) = reader.label();
	}
	cv::Mat trainingDataMat(data.nColumns(), data.nRows(), CV_32FC1, data.begin());
	trainSVM(trainingDataMat, labels, reader.nClasses());
}

void Svm::classify() {
	Features::LabeledSequenceFeatureReader reader;
	reader.initialize();

	featureQuantizer_.loadModel();

	u32 nClassificationErrors = 0;

	/* create container with all SVMs */
	std::vector<CvSVM*> svms;
	for (u32 c = 0; c < reader.nClasses(); c++) {
		svms.push_back(new CvSVM());
		std::stringstream filename;
		filename << modelDirectory_ << "/svm-" << c;
		svms.at(c)->load(filename.str().c_str());
	}

	/* classify each video */
	for (u32 i = 0; i < reader.totalNumberOfSequences(); i++) {
		Math::Vector<Float> f;
		featureQuantizer_.quantize(reader.next(), f);
		cv::Mat data(1, f.nRows(), CV_32FC1, f.begin());
		// prediction with all SVMs
		Math::Vector<Float> scores(reader.nClasses());
		for (u32 c = 0; c < reader.nClasses(); c++) {
			// there is a sign switch in the OpenCV implementation, so we need to use the negative score
			scores.at(c) = -svms.at(c)->predict(data, true);
		}
		nClassificationErrors += (scores.argMax() == reader.label() ? 0 : 1);
	}

	/* clean up */
	for (u32 c = 0; c < reader.nClasses(); c++)
		delete svms.at(c);

	/* log accuracy */
	Core::Log::openTag("accuracy");
	Core::Log::os() << 1.0 - (Float)nClassificationErrors / reader.totalNumberOfSequences();
	Core::Log::closeTag();
}
