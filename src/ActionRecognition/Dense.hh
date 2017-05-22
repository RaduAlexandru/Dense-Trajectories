#ifndef DENSE_HH
#define DENSE_HH

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>


#include "Track.hh"
#include "ArrayFastest.hh"
#include "Histogram.hh"
#include <fstream>

namespace ActionRecognition {
class Dense
{
private:
  // used for efficient access to cv::Mat
  struct MemoryAccessor {
    u32 rows, cols;
    Float* mem;
    MemoryAccessor(u32 r, u32 c, Float* m) { rows = r; cols = c; mem = m; }
    Float operator()(u32 y, u32 x) const  { return mem[y * cols + x]; }
  };
    static const Core::ParameterString paramVideoList_;




    std::string videoList_;

    typedef std::vector<cv::Mat> Video;



    cv::Mat mat2gray(const cv::Mat& src);
    void showVideo(const Video& vid);
    void readVideo(const std::string& filename, Video& result);

    void opticalFlow(const Video& video, Video& flow, Video& flowAngle, Video& flowMag);
    void makeTracks(std::vector<Track>& tracks, std::vector<cv::Point>&  points, int start_time);
    void track(std::vector<Track>& tracks, Video& flow);
    // void extractFeatures(const std::string& filename);
    void extractTrajectories(Video& video, Video& flow, std::vector<Track>& tracks);
    void denseSample(cv::Mat frame, std::vector<cv::Point>& points,  int stepSize);
    void filterTracks(std::vector<Track>& tracks);
    void writeTracksToFile(std::string trackFile, std::vector<Track> tracks);
    void derivatives(const Video& in, Video& Lx, Video& Ly, Video& Lt);
    void compute_grad_orientations_magnitudes(Video Lx, Video Ly, Video& grad_mags, Video& grad_orientations );
    void compute_mbh(Video flow, Video& mbh_x_mag, Video& mbh_x_orientation, Video& mbh_y_mag, Video& mbh_y_orientation);
    void computeDescriptors(Video& video,std::vector<Track>& tracks, Video Lx, Video Ly, Video flow, Video flowAngle, Video flowMag);
    int write_descriptors_to_file(std::vector<Track> tracks, std::ofstream& file);

public:
    Dense();
    virtual ~Dense();
    void run();
};

}
#endif // DENSE_HH
