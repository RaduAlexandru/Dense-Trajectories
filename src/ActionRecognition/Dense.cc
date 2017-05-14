#include "Dense.hh"
#include <algorithm>

using namespace ActionRecognition;

const Core::ParameterString Dense::paramVideoList_("video-list", "", "dense");

// constructor
Dense::Dense():
    videoList_(Core::Configuration::config(paramVideoList_))
{}


// empty destructor
Dense::~Dense()
{}


void Dense::readVideo(const std::string& filename, Video& result) {
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


void Dense::extractFeatures(const std::string& filename) {
    Video video;
    readVideo(filename, video);

    pyramidVideo pyrVideo;
    u_int scalNo = 4;
    for (u_int i = 0 ; i < video.size();i++){
        pyramidFrame pyrF ;

        cv::Mat dst ,prev;
        float downsize = 1.0 /sqrt(2.0);
        pyrF.push_back(video[i].clone());
        prev =  pyrF[0];

        for (int j = 1 ; j < scalNo ; j++){
            cv:: resize( prev, dst, cv::Size( prev.cols * downsize, prev.rows*downsize ) );
            pyrF.push_back(dst.clone());
            prev=dst;
        }
        pyrVideo.push_back(pyrF);
    }


    for (u_int s = 0 ; s < scalNo ; s++){
        for (u_int i = 0 ; i < video.size();i++){
            cv::imshow("",pyrVideo[i][s]);
            cv::waitKey(50);
        }
    }




}

void Dense::run() {
    if (videoList_.empty())
        Core::Error::msg("dense.video-list must not be empty.") << Core::Error::abort;

    Core::AsciiStream in(videoList_, std::ios::in);
    std::string filename;
    while (in.getline(filename)) {
        extractFeatures(filename);
    }


}
