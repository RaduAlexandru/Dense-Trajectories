#ifndef DENSE_HH
#define DENSE_HH

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace ActionRecognition {
class Dense
{
private:
    static const Core::ParameterString paramVideoList_;

    std::string videoList_;

    typedef std::vector<cv::Mat> Video;
    typedef std::vector<cv::Mat> pyramidFrame;
    typedef std::vector<std::vector<cv::Mat> > pyramidVideo;

    void readVideo(const std::string& filename, Video& result);
    void extractFeatures(const std::string& filename);

public:
    Dense();
    virtual ~Dense();
    void run();
};

}
#endif // DENSE_HH
