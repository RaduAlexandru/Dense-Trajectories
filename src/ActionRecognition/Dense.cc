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

      //The tracks sometime get ut of the video for some reason
      if (new_point.x<0 || new_point.x>flow[0].cols){
        continue;
      }
      if (new_point.y<0 || new_point.x>flow[0].rows){
        continue;
      }

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

  //remove the ones that don't have a length of 15
  std::vector<Track>::iterator it;
  for(it = tracks.begin(); it != tracks.end();){
    if(it->getLength()<15){
      it = tracks.erase(it);
    }
    else{
      ++it;
    }
  }

  //remove the ones that are almost constant
  for(it = tracks.begin(); it != tracks.end();){
    int displacement=0;
    for (size_t p_id = 0; p_id < it->getLength()-1; p_id++) {
      int dis_x= std::fabs(it->getPoint(p_id).x - it->getPoint(p_id+1).x);
      int dis_y= std::fabs(it->getPoint(p_id).y - it->getPoint(p_id+1).y);
      displacement+=std::sqrt( dis_x*dis_x + dis_y*dis_y   );
    }
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


void Dense::derivatives(const Video& in, Video& Lx, Video& Ly, Video& Lt) {
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

void Dense::compute_grad_orientations_magnitudes(Video Lx, Video Ly, Video& grad_mags, Video& grad_orientations ){
  std::cout << "compute_grad_orientations_magnitudes" << '\n';

  grad_mags.clear();
  grad_orientations.clear();

  bool useDegree = true;    // use degree or rad
  grad_mags.resize(Lx.size());
  grad_orientations.resize(Lx.size());

  for (size_t i = 0; i < Lx.size(); i++) {
    // the range of the direction is [0,2pi) or [0, 360)
    cv::cartToPolar(Lx[i], Ly[i], grad_mags[i], grad_orientations[i], useDegree);
  }

}


void Dense::compute_mbh(Video flow, Video& mbh_x_mag, Video& mbh_x_orientation, Video& mbh_y_mag, Video& mbh_y_orientation){

  mbh_x_mag.resize(flow.size());
  mbh_x_orientation.resize(flow.size());
  mbh_y_mag.resize(flow.size());
  mbh_y_orientation.resize(flow.size());

  for (size_t i = 0; i < flow.size(); i++) {
    cv::Mat tmpXY[2];
    cv::split(flow[i], tmpXY);

    //mbh_x
    cv::Mat dx, dy;
    cv::Sobel(tmpXY[0], dx, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::Sobel(tmpXY[0], dy, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::cartToPolar(dx, dy, mbh_x_mag[i], mbh_x_orientation[i], true);


    //mbh_y
    cv::Sobel(tmpXY[1], dx, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::Sobel(tmpXY[1], dy, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::cartToPolar(dx, dy, mbh_y_mag[i], mbh_y_orientation[i], true);


  }



}


void Dense::computeDescriptors(Video& video,std::vector<Track>& tracks, Video Lx, Video Ly, Video flow, Video flowAngle, Video flowMag ){

  int vol_x_size=32;
  int vol_y_size=32;
  int vol_t_size=15;

  int cell_per_vol_x=2;
  int cell_per_vol_y=2;
  int cell_per_vol_t=3;

  int nbins_hog=8;
  float hist_range=360.0f;

  float mag_thresh_low=1e-8;

  int cell_size_x=std::ceil(vol_x_size/(float)cell_per_vol_x);
  int cell_size_y=std::ceil(vol_y_size/(float)cell_per_vol_y);
  int cell_size_t=std::ceil(vol_t_size/(float)cell_per_vol_t);

  // std::cout << "cell is " << cell_size_x << " " << cell_size_y << " " << cell_size_t << '\n';


  //compute gradient orientation and magnitude
  Video grad_mags, grad_orientations;
  Video mbh_x_mags, mbh_x_orientations, mbh_y_mags, mbh_y_orientations;
  compute_grad_orientations_magnitudes(Lx,Ly,grad_mags,grad_orientations);
  compute_mbh(flow,mbh_x_mags,mbh_x_orientations, mbh_y_mags, mbh_y_orientations);







  //get descriptor

  for (size_t i = 0; i < tracks.size(); i++) {
    Histogram  descriptor;
    std::vector<float> traj_desc= tracks[i].getDescriptor();
    descriptor.concatenate(traj_desc);

    //hog
    utils::Array<Histogram, 3> hist_hog_vol;
    size_t size_hog_vol [3]= { cell_per_vol_t, cell_per_vol_y, cell_per_vol_x }; // Array dimensions
    hist_hog_vol.resize(size_hog_vol,Histogram(nbins_hog, hist_range));

    //hof for normal flow
    utils::Array<Histogram, 3> hist_hof_vol;
    size_t size_hof_vol [3]= { cell_per_vol_t, cell_per_vol_y, cell_per_vol_x }; // Array dimensions
    hist_hof_vol.resize(size_hof_vol,Histogram(nbins_hog, hist_range));

    //hof with 1 bin for the bin that has low magnitude
    utils::Array<Histogram, 3> hist_hof_low_mag_vol;
    hist_hof_low_mag_vol.resize(size_hof_vol,Histogram(1, 360.0f));

    //mbh_x
    utils::Array<Histogram, 3> hist_mbhx_vol;
    hist_mbhx_vol.resize(size_hog_vol,Histogram(nbins_hog, hist_range));


    //mbh_y
    utils::Array<Histogram, 3> hist_mbhy_vol;
    hist_mbhy_vol.resize(size_hog_vol,Histogram(nbins_hog, hist_range));




    for (size_t t = 0; t < tracks[i].getLength()-1; t++) {
      for (size_t y = std::max(0,tracks[i].getPoint(t).y-16); y < std::min(video[0].rows,tracks[i].getPoint(t).y+15); y++) {
        for (size_t x = std::max(0,tracks[i].getPoint(t).x-16); x < std::min(video[0].cols,tracks[i].getPoint(t).x+15); x++) {
          //get which cell of the 16x16x5 volume does this pixel belong to
          int cell_idx_x= (x- (tracks[i].getPoint(t).x - vol_x_size/2 ) )/cell_size_x;
          int cell_idx_y= (y- (tracks[i].getPoint(t).y - vol_y_size/2 ) )/cell_size_y;
          int cell_idx_t= t/cell_size_t;
          // std::cout << "cell indexing i" << cell_idx_x << " " << cell_idx_y << " " << cell_idx_t << '\n';

          // if (cell_idx_x>=cell_per_vol_x){
          //   std::cout << "ERROR: cell_idx_x out of bounds" << '\n';
          //   std::cout << "cell_idx_x: " << cell_idx_x << '\n';
          //   std::cout << "point x is " << tracks[i].getPoint(t).x << '\n';
          //   std::cout << "x is " << x << '\n';
          //   std::cout << "vol_x_size is" << vol_x_size << '\n';
          //   std::cout << "cap of x is " << std::min(video[0].cols,tracks[i].getPoint(t).x+16) << '\n';
          //   std::cout << "vidoe cols is " << video[0].cols << '\n';
          //   std::cout << "track x +16 is "  << tracks[i].getPoint(t).x+16 << '\n';
          // }
          // if (cell_idx_y>=cell_per_vol_y){
          //   std::cout << "ERROR: cell_idx_y out of bounds" << '\n';
          //   std::cout << "cell_idx_y: " << cell_idx_y << '\n';
          //   std::cout << "point x is " << tracks[i].getPoint(t).y << '\n';
          //   std::cout << "y is " << y << '\n';
          //   std::cout << "vol_y_size is" << vol_y_size << '\n';
          // }
          // if (cell_idx_t>=cell_per_vol_t){
          //   std::cout << "ERROR: cell_idx_t out of bounds" << '\n';
          //   std::cout << "cell_idx_t: " << cell_idx_t << '\n';
          //   std::cout << "point t is " << tracks[i].getLength() << '\n';
          //   std::cout << "t is " << t << '\n';
          //   std::cout << "vol_t_size is" << vol_t_size << '\n';
          // }


          //we need it because the idx in time needs to be the time of the point + the time when the track started
          int t_idx=t+tracks[i].getStartTime();


          float grad_mag=grad_mags[t_idx].at<float>(y,x);
          float grad_orientation=grad_orientations[t_idx].at<float>(y,x);

          float flow_mag=flowMag[t_idx].at<float>(y,x);
          float flow_orientation=flowAngle[t_idx].at<float>(y,x);

          float mbh_x_mag=mbh_x_mags[t_idx].at<float>(y,x);
          float mbh_x_orientation=mbh_x_orientations[t_idx].at<float>(y,x);

          float mbh_y_mag=mbh_y_mags[t_idx].at<float>(y,x);
          float mbh_y_orientation=mbh_y_orientations[t_idx].at<float>(y,x);



          //hog
          hist_hog_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(grad_orientation,grad_mag);

          //hof
          if (flow_mag<mag_thresh_low) {
            hist_hof_low_mag_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(flow_orientation,flow_mag);
          }else{
            hist_hof_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(flow_orientation,flow_mag);
          }


          //mbh_x
          hist_mbhx_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(mbh_x_orientation,mbh_x_mag);


          //mbh_y
          hist_mbhy_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(mbh_y_orientation,mbh_y_mag);





        }
      }
    }



    //concatenate the the low threshold with the high threshold HOF
    for (size_t t = 0; t < cell_per_vol_t; t++) {
      for (size_t y = 0; y < cell_per_vol_y; y++) {
        for (size_t x = 0; x < cell_per_vol_x; x++) {
          hist_hof_vol[t][y][x].concatenate(hist_hof_low_mag_vol[t][y][x]);
        }
      }
    }


    Histogram hof_full;
    for(auto hist: hist_hof_vol){
      hof_full.concatenate(hist);
    }
    hof_full.normalize();


    Histogram hog_full;
    for(auto hist: hist_hog_vol){
      hog_full.concatenate(hist);
    }
    hog_full.normalize();


    Histogram mbhx_full;
    for(auto hist: hist_mbhx_vol){
      mbhx_full.concatenate(hist);
    }
    mbhx_full.normalize();


    Histogram mbhy_full;
    for(auto hist: hist_mbhy_vol){
      mbhy_full.concatenate(hist);
    }
    mbhy_full.normalize();

    //full descriptor
    descriptor.concatenate(hog_full);
    descriptor.concatenate(hof_full);
    descriptor.concatenate(mbhx_full);
    descriptor.concatenate(mbhy_full);

    tracks[i].descriptor=descriptor;

    // std::cout << "descriptor has size " << descriptor.size() << '\n';
    // if(i==0){
    //   std::cout << "descripor is" << descriptor.to_string() << '\n';
    // }



  }

  // grad_mags.clear();
  // grad_orientations.clear();
  // mbh_x_mags.clear();
  // mbh_x_orientations.clear();
  // mbh_y_mags.clear();
  // mbh_y_orientations.clear();

}

int Dense::write_descriptors_to_file(std::vector<Track> tracks, std::ofstream& file){

  int descriptor_written=0;
  for (size_t i = 0; i < tracks.size(); i++) {
    //if the descriptor is not yet initialized it means it's empty
    if (!tracks[i].descriptor.to_string().empty()  ){
      file << tracks[i].descriptor.to_string() << std::endl;
      descriptor_written++;
    }
  }

  return descriptor_written;


}


void Dense::run() {
    if (videoList_.empty())
        Core::Error::msg("dense.video-list must not be empty.") << Core::Error::abort;

    // Core::AsciiStream in(videoList_, std::ios::in);
    // std::string filename;
    //
    // std::string descriptor_file_path = "./desc.txt";
    // std::ofstream desc_file;
    // desc_file.open (descriptor_file_path);
    //
    // int nr_vectors=0;
    // int vector_dimensions=0;
    // int nr_videos=0;
    //
    //
    // while (in.getline(filename)) {
    //   std::cout << "video  " << filename<< '\n';
    //
    //     Video video;
    //     Video flow, flowAngle, flowMag;
    //     Video Lx,Ly,Lt;
    //     std::vector<Track> tracks;
    //     readVideo(filename, video);
    //     // showVideo(video);
    //
    //     //compute_optical_flow
    //     opticalFlow(video, flow, flowAngle, flowMag);
    //     extractTrajectories(video, flow, tracks);
    //     filterTracks(tracks);
    //
    //     // std::string trackFile="./tracks.txt";
    //     // writeTracksToFile(trackFile,tracks);
    //
    //     //derivatives
    //     derivatives(video,Lx,Ly,Lt);
    //     computeDescriptors(video,tracks, Lx, Ly, flow, flowAngle, flowMag);
    //
    //
    //     //write to file
    //     int descriptor_written=write_descriptors_to_file(tracks, desc_file);
    //     desc_file << "#"<< std::endl;
    //
    //     nr_vectors+=descriptor_written;
    //     vector_dimensions=426;  //TODO remove hardcode
    //     nr_videos++;
    //
    //     // extractTexturedPoints(video);
    //     // extractFeatures(filename);
    //     video.clear();
    //     flow.clear();
    //     flowAngle.clear();
    //     flowMag.clear();
    //
    //     Lx.clear();
    //     Ly.clear();
    //     Lt.clear();
    //     tracks.clear();
    //
    //     std::cout << "finished" << '\n';
    // }
    //
    // //Add header to file
    // desc_file.seekp(0); //Move at start of file
    // desc_file << nr_vectors << " " << vector_dimensions << " " << nr_videos << std::endl;
    // desc_file.close();


    //Read the features back again and do pca on them
    std::string descriptor_file_path = "./desc.txt";


}
