/*
 * Stip.hh
 *
 *  Created on: Apr 25, 2017
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_STIP_HH_
#define ACTIONRECOGNITION_STIP_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace ActionRecognition {

class Stip
{
private:
	struct Point {
		u32 x, y, t;
		Point(u32 _x, u32 _y, u32 _t) { x = _x; y = _y; t = _t; }
	};
	struct InterestPoint : Point {
		Float score;
		InterestPoint(u32 _x, u32 _y, u32 _t, Float _score) : Point(_x, _y, _t) { score = _score; }
	};
	// used for efficient access to cv::Mat
	struct MemoryAccessor {
		u32 rows, cols;
		Float* mem;
		MemoryAccessor(u32 r, u32 c, Float* m) { rows = r; cols = c; mem = m; }
		Float operator()(u32 x, u32 y) const  { return mem[x * cols + y]; }
	};
	// compare the scores of two interest points (for std::sort)
	static bool compareInterestPoints(const InterestPoint i, const InterestPoint j) { return i.score > j.score; }

	static const Core::ParameterFloat paramInterestValue_;
	static const Core::ParameterFloat paramKeepRatio_;
	static const Core::ParameterFloatList paramSpatialScales_;
	static const Core::ParameterFloatList paramTemporalScales_;
	static const Core::ParameterString paramVideoList_;

	typedef std::vector<cv::Mat> Video;

	Float interestValue_;
	Float keepRatio_;
	std::vector<Float> spatialScales_;
	std::vector<Float> temporalScales_;
	std::string videoList_;

	std::vector<InterestPoint> interestPoints_;
	std::vector< Math::Vector<Float> > hoghof_;

	void show(const Video& video);
	void readVideo(const std::string& filename, Video& result);
	void opticalFlow(const Video& video, Video& flowAngle, Video& flowMag);
	void filter3d(const Video& in, Video& out, Float sigma, Float tau);
	void derivatives(const Video& in, Video& Lx, Video& Ly, Video& Lt);
	void interestPoints(const Video& Lx, const Video& Ly, const Video& Lt, Float sigma, Float tau);
	void createHistogram(const Video& gradAngle, const Video& gradMag, const Video& flowAngle, const Video& flowMag,
			const InterestPoint& p, Float sigma, Float tau);
	void extractFeatures(const std::string& filename);
public:
	Stip();
	virtual ~Stip();
	void run();
};

} // namespace


#endif /* ACTIONRECOGNITION_STIP_HH_ */
