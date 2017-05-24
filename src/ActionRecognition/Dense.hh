#ifndef DENSE_HH
#define DENSE_HH

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml/ml.hpp>


#include "Track.hh"
#include "ArrayFastest.hh"
#include "Histogram.hh"


#include "Gmm.hh"

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

    void read_features_from_file_Mat(std::string descriptor_file_path, cv::Mat& features,  std::vector<int>& nr_features_per_video);
    void read_features_from_file_math(std::string descriptor_file_path, Math::Matrix<Float>& features);
    void read_features_per_video_from_file_Mat(std::string descriptor_file_path, std::vector<cv::Mat >& features_per_video, int max_nr_videos);
    void read_features_per_video_from_file_math(std::string descriptor_file_path, std::vector<Math::Matrix<Float> >& features_per_video, int max_nr_videos);
    cv::PCA compressPCA(const cv::Mat& pcaset, int maxComponents, const cv::Mat& testset, cv::Mat& compressed);
    void write_nr_features_per_video_to_file(std::string nr_features_per_video_file_path, std::vector<int> nr_features_per_video);
    void write_compressed_features_to_file(std::string desc_compressed_file_path, cv::Mat feat_compressed, std::vector<int> nr_features_per_video);


    Float calculateDeterminenet(const Math::Vector<Float>& sigma);
    void calculateInverse(const Math::Vector<Float>& diagonalMat, Math::Vector<Float>& result);
    void computeFisherVectors(Math::Matrix<Float>& fisher_vectors, std::vector <Math::Matrix<Float> >&  features_per_video, const Math::Matrix<Float>& means, const Math::Vector<Float>& weights, const Math::Matrix<Float>& sigmas);
    float u(Math::Vector<Float>& x, const Math::Matrix<Float>& means, const Math::Matrix<Float>& sigmas, int k  );

    void task_1_2_extract_trajectories(std::string descriptor_file_path);
    void task_3_pca(std::string descriptor_file_path, std::string desc_compressed_file_path);
    void task_3_gmm(std::string desc_compressed_file_path);
    void task_3_fisher(std::string desc_compressed_file_path);

public:
    Dense();
    virtual ~Dense();
    void run();
};

}
#endif // DENSE_HH
