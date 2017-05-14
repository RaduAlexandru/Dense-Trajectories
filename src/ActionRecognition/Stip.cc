/*
 * Stip.cc
 *
 *  Created on: Apr 25, 2017
 *      Author: richard
 */

#include "Stip.hh"
#include <algorithm>
#include "Features/FeatureWriter.hh"

using namespace ActionRecognition;

const Core::ParameterFloat Stip::paramInterestValue_("interest-value", 0.005, "stip");

const Core::ParameterFloat Stip::paramKeepRatio_("keep-ratio", 0.5, "stip");

const Core::ParameterFloatList Stip::paramSpatialScales_("spatial-scales", "1.0, 2.0, 3.0", "stip");

const Core::ParameterFloatList Stip::paramTemporalScales_("temporal-scales", "1.0, 2.0", "stip");

const Core::ParameterString Stip::paramVideoList_("video-list", "", "stip");

// constructor
Stip::Stip() :
		interestValue_(Core::Configuration::config(paramInterestValue_)),
		keepRatio_(Core::Configuration::config(paramKeepRatio_)),
		spatialScales_(Core::Configuration::config(paramSpatialScales_)),
		temporalScales_(Core::Configuration::config(paramTemporalScales_)),
		videoList_(Core::Configuration::config(paramVideoList_))
{}

// empty destructor
Stip::~Stip()
{}

void Stip::show(const Video& vid) {
	Video video = vid;
	for (u32 i = 0; i < interestPoints_.size(); i++) {
		// switch x and y because Point is (width, height) and (x, y) is for (rows, cols)
		cv::Point p1(interestPoints_.at(i).y-1, interestPoints_.at(i).x-1);
		cv::Point p2(interestPoints_.at(i).y+1, interestPoints_.at(i).x+1);
		cv::rectangle(video.at(interestPoints_.at(i).t), p1, p2, 0.5, 1);
	}
	for (u32 t = 0; t < video.size(); t++) {
		cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
		cv::imshow("Display window", video.at(t));
		cv::waitKey(0);
	}
}

void Stip::readVideo(const std::string& filename, Video& result) {
	// open video file
	cv::VideoCapture capture(filename);
	if(!capture.isOpened())
		Core::Error::msg() << "Unable to open Video: " << filename << Core::Error::abort;
	cv::Mat frame, tmp;
	result.clear();
	// read all frames
	u32 nFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
	while ((nFrames > 0) && (capture.read(frame))) {
		if (frame.channels() == 3)
			cv::cvtColor(frame, tmp, CV_BGR2GRAY);
		else
			tmp = frame;
		result.push_back(cv::Mat());
		tmp.convertTo(result.back(), CV_32FC1, 1.0/255.0);
		nFrames--;
	}
	capture.release();
}

void Stip::opticalFlow(const Video& video, Video& flowAngle, Video& flowMag) {
	flowAngle.clear();
	flowMag.clear();
	for (u32 t = 0; t < video.size() - 1; t++) {
		cv::Mat tmpFlow;
		cv::Mat tmpXY[2];
		cv::calcOpticalFlowFarneback(video.at(t), video.at(t+1), tmpFlow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
		cv::split(tmpFlow, tmpXY);
		cv::Mat magnitude, angle;
		cv::cartToPolar(tmpXY[0], tmpXY[1], magnitude, angle, true);
		flowAngle.push_back(angle);
		flowMag.push_back(magnitude);
	}
}

void Stip::filter3d(const Video& in, Video& out, Float sigma, Float tau) {
	out.clear();
	// filter in spatial domain
	Video tmp(in.size());
	for (u32 t = 0; t < in.size(); t++) {
		cv::GaussianBlur(in.at(t), tmp.at(t), cv::Size(0,0), sigma, sigma);
	}
	// filter in temporal domain
	u32 ksize = u32(3.0 * tau);
	cv::Mat kernel = cv::getGaussianKernel(ksize*2 + 1, tau, CV_32F);
	for (u32 t = 0; t < in.size(); t++) {
		out.push_back(cv::Mat(tmp.at(t).rows, tmp.at(t).cols, tmp.at(t).type(), 0.0));
		Float* ptr1 = (Float*) out.back().data;
		// add weighted neighboring frames
		for (s32 w = -ksize; w <= (s32)ksize; w++) {
			u32 i = std::min(std::max(0, (s32)t+w), (s32)tmp.size() - 1);
			Float* ptr2 = (Float*) tmp.at(i).data;
			Math::axpy(tmp.at(t).rows * tmp.at(t).cols, kernel.at<Float>(w+ksize), ptr2, 1, ptr1, 1);
		}
	}
}

void Stip::derivatives(const Video& in, Video& Lx, Video& Ly, Video& Lt) {
	Lx.resize(in.size() - 1);
	Ly.resize(in.size() - 1);
	Lt.resize(in.size() - 1);
	/* loop over the original frames */
	for (u32 t = 0; t < in.size() - 1; t++) {
		cv::Sobel(in.at(t), Lx.at(t), CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::Sobel(in.at(t), Ly.at(t), CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
		Lt.at(t) = in.at(t) - in.at(t+1);
	}
}

void Stip::interestPoints(const Video& Lx, const Video& Ly, const Video& Lt, Float sigma, Float tau) {
	/* compute elements of second moment matrix */
	Video Lxx, Lyy, Ltt, Lxy, Lxt, Lyt;
	for (u32 t = 0; t < Lx.size(); t++) {
		Lxx.push_back(Lx.at(t).mul(Lx.at(t)));
		Lyy.push_back(Ly.at(t).mul(Ly.at(t)));
		Ltt.push_back(Lt.at(t).mul(Lt.at(t)));
		Lxy.push_back(Lx.at(t).mul(Ly.at(t)));
		Lxt.push_back(Lx.at(t).mul(Lt.at(t)));
		Lyt.push_back(Ly.at(t).mul(Lt.at(t)));
	}
	/* apply spatio-temporal Gaussian filter */
	Video tmp;
	filter3d(Lxx, tmp, sigma, tau); Lxx.swap(tmp);
	filter3d(Lyy, tmp, sigma, tau); Lyy.swap(tmp);
	filter3d(Ltt, tmp, sigma, tau); Ltt.swap(tmp);
	filter3d(Lxy, tmp, sigma, tau); Lxy.swap(tmp);
	filter3d(Lxt, tmp, sigma, tau); Lxt.swap(tmp);
	filter3d(Lyt, tmp, sigma, tau); Lyt.swap(tmp);

	/* compute 3D Harris function */
	Video H(Lx.size());
	for (u32 t = 0; t < H.size(); t++) {
		cv::Mat det =
				Lxx.at(t).mul(Lyy.at(t).mul(Ltt.at(t))) +
				2 * Lxy.at(t).mul(Lyt.at(t).mul(Lxt.at(t))) -
				Lxt.at(t).mul(Lxt.at(t).mul(Lyy.at(t))) -
				Lyt.at(t).mul(Lyt.at(t).mul(Lxx.at(t))) -
				Lxy.at(t).mul(Lxy.at(t).mul(Ltt.at(t)));
		cv::Mat trace = Lxx.at(t) + Lyy.at(t) + Ltt.at(t);
		cv::pow(trace, 3.0, trace);
		H.at(t) = det - interestValue_ * trace;
	}
	/* find local maxima (border frames/border pixels are excluded) */
	for (u32 t = 1; t < H.size()-1; t++) {
		MemoryAccessor p (H.at(t  ).rows, H.at(t  ).cols, (Float*) H.at(t  ).data);
		MemoryAccessor pL(H.at(t-1).rows, H.at(t-1).cols, (Float*) H.at(t-1).data);
		MemoryAccessor pR(H.at(t+1).rows, H.at(t+1).cols, (Float*) H.at(t+1).data);
		for (u32 x = 1; x < p.rows - 1; x++) {
			for (u32 y = 1; y < p.cols - 1; y++) {
				bool isInteresting = (p(x,y) > 0);
				if (!isInteresting) continue;
				// check all 3d neighbors
				for (s32 dx = -1; dx <= 1; dx++)
					for (s32 dy = -1; dy <= 1; dy++)
						if ( (p(x+dx,y+dy) > p(x,y)) || (pL(x+dx,y+dy) > p(x,y)) || (pR(x+dx,y+dy) > p(x,y)) ) isInteresting = false;
				if (isInteresting) interestPoints_.push_back(InterestPoint(x, y, t, p(x,y)));
			}
		}
	}
	/* keep only a fraction of keepRatio_ interest points from this scale */
	std::sort(interestPoints_.begin(), interestPoints_.end(), compareInterestPoints);
	interestPoints_.erase(interestPoints_.begin() + (interestPoints_.size() * keepRatio_), interestPoints_.end());
}

void Stip::createHistogram(const Video& gradAngle, const Video& gradMag, const Video& flowAngle, const Video& flowMag,
		const InterestPoint& p, Float sigma, Float tau) {
	// initialize histograms with zero
	std::vector< Math::Vector<Float> > histograms;
	for (u32 h = 0; h < 3*3*2*2; h++) { // 3x3x2 hog and hof histograms
		histograms.push_back(Math::Vector<Float>(4)); // use four bins per histogram
		histograms.back().setToZero();
	}
	// define volume around interest point
	u32 k = 9;
	s32 spSize = k * sigma;
	s32 tpSize = k * tau;
	// go over all pixels in the volume
	for (u32 t = std::max(0, (s32)p.t - tpSize); t < std::min((u32)gradAngle.size(), p.t + tpSize); t++) {
		MemoryAccessor angleHog(gradAngle.at(t).rows, gradAngle.at(t).cols, (Float*)gradAngle.at(t).data);
		MemoryAccessor magHog(gradMag.at(t).rows, gradMag.at(t).cols, (Float*)gradMag.at(t).data);
		MemoryAccessor angleHof(flowAngle.at(t).rows, flowAngle.at(t).cols, (Float*)flowAngle.at(t).data);
		MemoryAccessor magHof(flowMag.at(t).rows, flowMag.at(t).cols, (Float*)flowMag.at(t).data);
		for (u32 x = std::max(0, (s32)p.x - spSize); x < std::min((u32)gradAngle.at(t).rows, p.x + spSize); x++) {
			for (u32 y = std::max(0, (s32)p.y - spSize); y < std::min((u32)gradAngle.at(t).rows, p.y + spSize); y++) {
				// compute histogram index of current pixel (x,y,t) in the 3x3x2 grid
				u32 histIdx = 0;
				if (t > p.t) histIdx += 9;
				if (x > (Float)p.x - (1.0 / 3.0)*spSize) histIdx += 3;
				if (x > (Float)p.x + (1.0 / 3.0)*spSize) histIdx += 3;
				if (y > (Float)p.y - (1.0 / 3.0)*spSize) histIdx += 1;
				if (y > (Float)p.y + (1.0 / 3.0)*spSize) histIdx += 1;
				// compute bin index in 144-dimensional array (3*3*2*4*2)
				u32 binIdx = u32(angleHog(x,y) / 45.1) % 4; // 45° leads to 8 bins, % 4: mirror to make sign independent
				histograms.at(histIdx).at(binIdx) += magHog(x,y);
				binIdx = u32(angleHof(x,y) / 45.1) % 4; // 45° leads to 8 bins, % 4: mirror to make sign independent
				histograms.at(histIdx + 3*3*2).at(binIdx) += magHof(x,y);
			}
		}
	}
	// normalize all histograms
	for (u32 h = 0; h < histograms.size(); h++) {
		if (histograms.at(h).asum() > 0)
			histograms.at(h).scale(1.0 / histograms.at(h).asum());
	}
	// combine everything in result vector
	Math::Vector<Float> result(3*3*2*4*2);
	for (u32 h = 0; h < histograms.size(); h++) {
		for (u32 bin = 0; bin < 4; bin++) {
			result.at(h * 4 + bin) = histograms.at(h).at(bin);
		}
	}
	// add if vector is nonzero
	if (result.asum() > 0)
		hoghof_.push_back(result);
}

void Stip::extractFeatures(const std::string& filename) {
	Video video;
	/* read the video */
	readVideo(filename, video);
	/* extract optical flow (angles of flow vectors) */
	Video flowAngle, flowMag;
	opticalFlow(video, flowAngle, flowMag);
	/* loop over different scales */
	for (u32 sp = 0; sp < spatialScales_.size(); sp++) {
		for (u32 tp = 0; tp < temporalScales_.size(); tp++) {
			/* extract interest points */
			Video filteredVid, Lx, Ly, Lt;
			filter3d(video, filteredVid, spatialScales_.at(sp), temporalScales_.at(tp));
			derivatives(filteredVid, Lx, Ly, Lt);
			interestPoints_.clear();
			interestPoints(Lx, Ly, Lt, 2*spatialScales_.at(sp), 2*temporalScales_.at(tp));
			/* compute HoG and HoF features for interest points */
			Video gradAngle, gradMag;
			for (u32 t = 0; t < Lx.size(); t++) {
				cv::Mat magnitude, angle;
				cv::cartToPolar(Lx.at(t), Ly.at(t), magnitude, angle, true);
				gradAngle.push_back(angle);
				gradMag.push_back(magnitude);
			}
			/* compute hog/hof descriptors */
			for (u32 i = 0; i < interestPoints_.size(); i++) {
				createHistogram(gradAngle, gradMag, flowAngle, flowMag, interestPoints_.at(i) , spatialScales_.at(sp), temporalScales_.at(tp));
			}
		}
	}
}

void Stip::run() {

	if (videoList_.empty())
		Core::Error::msg("stip.video-list must not be empty.") << Core::Error::abort;

	Core::AsciiStream in(videoList_, std::ios::in);
	std::string video;
	std::vector< Math::Matrix<Float> > features;
	u32 totalNumberOfVectors = 0;

	/* extract features for each video */
	while (in.getline(video)) {
		hoghof_.clear();
		extractFeatures(video);
		features.push_back(Math::Matrix<Float>(3*3*2*4*2, hoghof_.size()));
		for (u32 i = 0; i < hoghof_.size(); i++)
			Math::copy(hoghof_.at(i).size(), hoghof_.at(i).begin(), 1, features.back().begin() + hoghof_.at(i).size() * i, 1);
		totalNumberOfVectors += hoghof_.size();
	}

	/* write extracted features to an output file */
	Features::SequenceFeatureWriter writer;
	writer.initialize(totalNumberOfVectors, 3*3*2*4*2, features.size());
	for (u32 i = 0; i < features.size(); i++)
		writer.write(features.at(i));

}
