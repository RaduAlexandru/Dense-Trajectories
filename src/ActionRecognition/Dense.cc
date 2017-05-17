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

void Dense::showVideo(const Video& vid) {
	Video video = vid;
	for (u32 t = 0; t < video.size(); t++) {
		cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
		cv::imshow("Display window", video.at(t));
		cv::waitKey(0);
	}
}

cv::Mat Dense::mat2gray(const cv::Mat& src){
    cv::Mat dst;
    cv::normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

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

void Dense::opticalFlow(const Video& video, Video& flow, Video& flowAngle, Video& flowMag) {
  flow.clear();
	flowAngle.clear();
	flowMag.clear();
	for (u32 t = 0; t < video.size() - 1; t++) {
    cv::Mat tmpFlow;
		cv::Mat tmpXY[2];

    cv::normalize(video[t], video[t], 0.0f, 255.0f, cv::NORM_MINMAX, CV_32FC1);
    cv::normalize(video[t+1], video[t+1], 0.0f, 255.0f, cv::NORM_MINMAX, CV_32FC1);
		// cv::calcOpticalFlowFarneback(video.at(t), video.at(t+1), tmpFlow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
    cv::calcOpticalFlowFarneback(video.at(t),  video.at(t+1), tmpFlow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cv::normalize(video[t], video[t], 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);
    cv::normalize(video[t+1], video[t+1], 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);


		cv::split(tmpFlow, tmpXY);
		cv::Mat magnitude, angle;
		cv::cartToPolar(tmpXY[0], tmpXY[1], magnitude, angle, true);
		flowAngle.push_back(angle);
		flowMag.push_back(magnitude);



    //median filter it
    cv::Mat flowChannels[2];
	  cv::split(tmpFlow, flowChannels);
	  cv::medianBlur(flowChannels[0], flowChannels[0], 5);
	  cv::medianBlur(flowChannels[1], flowChannels[1], 5);
    cv::merge(flowChannels, 2, tmpFlow);



    flow.push_back(tmpFlow);




    //find min and maxs
    // double min, max;
    // cv::minMaxLoc(tmpFlow, &min, &max);
    // std::cout << "flow at " << t  << "minmax" << min << " " << max << '\n';
    // cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    // // cv::imshow("Display window", mat2gray(minEigenvalMat));
    // cv::imshow("Display window", mat2gray(tmpFlow));
    // cv::waitKey(0);


	}
}

void Dense::makeTracks(std::vector<Track>& tracks, std::vector<cv::Point>&  points, int start_time){
  for (size_t i = 0; i < points.size(); i++) {
    tracks.push_back( Track( points[i], start_time )   );
  }
}

void Dense::track(std::vector<Track>& tracks, Video& flow){

  // std::cout << "tracking" << '\n';

  for (size_t i = 0; i < tracks.size(); i++) {
    if (tracks[i].getLength()<15){

      cv::Point new_point;
      int x=tracks[i].getLastPoint().x;
      int y=tracks[i].getLastPoint().y;
      int t=tracks[i].getLastTime();

      // std::cout << " prev point is " << i << " " <<x << " " << y << '\n';
      // std::cout << "flow has size " << flow.size() << '\n';
      // std::cout << " accesing at time " << t << '\n';

			// new_point.x = tracks[i].getLastPoint().x + flow[t].ptr<float>(y)[2*x];
      // new_point.y = tracks[i].getLastPoint().y + flow[t].ptr<float>(y)[2*x+1];
      new_point.x = tracks[i].getLastPoint().x + flow[t].at<cv::Point2f>(y,x).x;
      new_point.y = tracks[i].getLastPoint().y + flow[t].at<cv::Point2f>(y,x).y;
      // std::cout << "flow" << flow[t].at<cv::Point2f>(y,x).x << " " << flow[t].at<cv::Point2f>(y,x).y << '\n';

      tracks[i].addPoint(new_point);

      // std::cout << " new point is " << i << " " << new_point.x << " " << new_point.y << '\n';

    }
  }


  // for (size_t i = 0; i < flow[0].rows; i++) {
  //   for (size_t j = 0; j < flow[0].cols; j++) {
  //     std::cout << "flow" << flow[2].at<cv::Point2f>(i,j).x << " " << flow[2].at<cv::Point2f>(i,j).y << '\n';
  //   }
  // }

}


void Dense::extractTrajectories(Video& video, Video& flow, std::vector<Track> & tracks){


  // std::vector<Track> tracks;
  tracks.clear();


  for (u_int i = 0 ; i < video.size()-1;i++){
    if (i==0){
      std::vector<cv::Point> points;
      denseSample(video[i],points,5);
      makeTracks(tracks,points,i);
    }


    //with some criteria sample again
    if (i%12==0){
      std::vector<cv::Point> points;
      denseSample(video[i],points,5);
      makeTracks(tracks,points,i);
    }




    track (tracks, flow);






  }

  //extract densly in th first frame
  //for each frame
    //if (some criteria){
    // ddensly sample
    //}
    //track the points



}


void Dense::denseSample(cv::Mat frame, std::vector<cv::Point>& points,  int stepSize){

  cv::Mat minEigenvalMat;
  cv::cornerMinEigenVal( frame, minEigenvalMat, 3, 3, cv::BORDER_DEFAULT );

  //get the threhold
  double maxEigenval = 0;
  cv::minMaxLoc(minEigenvalMat, 0, &maxEigenval);
  float threshold = maxEigenval*0.001;

  MemoryAccessor minEigenvalMat_acc (minEigenvalMat.rows, minEigenvalMat.cols, (Float*) minEigenvalMat.data);
  for (u32 x = 0; x < minEigenvalMat.cols-stepSize; x=x+stepSize) {
    for (u32 y = 0; y < minEigenvalMat.rows-stepSize; y=y+stepSize) {
      if (minEigenvalMat_acc(y,x)>threshold){
        points.push_back(cv::Point(x,y));
      }
    }
  }




  // //droaw the points on the frame
  // for (size_t i = 0; i < points.size(); i++) {
  //    cv::circle(frame, points[i], 3, cv::Scalar(0,0,255));
  // }
  //
  //
  //
  // cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  // // cv::imshow("Display window", mat2gray(minEigenvalMat));
  // cv::imshow("Display window", frame);
  // cv::waitKey(0);



}


void Dense::filterTracks(std::vector<Track>& tracks){

  std::vector<Track>::iterator it;
  // for(it = tracks.begin(); it != tracks.end();){
  //   if(it->getLength()<15){
  //     it = tracks.erase(it);
  //   }
  //   else{
  //     ++it;
  //   }
  // }

  for(it = tracks.begin(); it != tracks.end();){
    int displacement=0;
    for (size_t p_id = 0; p_id < it->getLength()-1; p_id++) {
      int dis_x= std::fabs(it->getPoint(p_id).x - it->getPoint(p_id+1).x);
      int dis_y= std::fabs(it->getPoint(p_id).y - it->getPoint(p_id+1).y);
      displacement+=std::sqrt( dis_x*dis_x + dis_y*dis_y   );
    }
    // std::cout << "displacement" << displacement << '\n';
    if (displacement<=12){
      it = tracks.erase(it);
    }
    else{
      ++it;
    }
  }




}


void Dense::writeTracksToFile(std::string trackFile, std::vector<Track> tracks){

  std::ofstream myfile;
  myfile.open (trackFile);
  myfile << "# X Y Z\n";


  for (size_t i = 0; i < tracks.size(); i++) {
    for (size_t p_id = 0; p_id < tracks[i].getLength(); p_id++) {
      myfile  << tracks[i].getPoint(p_id).x << " " << tracks[i].getPoint(p_id).y << " " << tracks[i].getStartTime()+p_id  << "\n";
    }
    myfile << "\n";
  }


  myfile.close();


}

void Dense::run() {
    if (videoList_.empty())
        Core::Error::msg("dense.video-list must not be empty.") << Core::Error::abort;

    Core::AsciiStream in(videoList_, std::ios::in);
    std::string filename;
    while (in.getline(filename)) {
        Video video;
        Video flow, flowAngle, flowMag;
        std::vector<Track> tracks;
        readVideo(filename, video);
        // showVideo(video);

        //compute_optical_flow
        opticalFlow(video, flow, flowAngle, flowMag);
        extractTrajectories(video, flow, tracks);
        filterTracks(tracks);

        // for (size_t i = 0; i < tracks.size(); i++) {
        //   /* code */
        // }

        std::string trackFile="./tracks.txt";
        writeTracksToFile(trackFile,tracks);

        // std::cout << "total nr of tracks" << tracks.size() << '\n';
        // for (size_t i = 0; i < tracks[0].getLength(); i++) {
        //   std::cout << "track" << tracks[0].getPoint(i).x  << " " << tracks[0].getPoint(i).y  << '\n';
        // }




        // extractTexturedPoints(video);
        // extractFeatures(filename);
        video.clear();
        flow.clear();
        flowAngle.clear();
        flowMag.clear();
    }


}
