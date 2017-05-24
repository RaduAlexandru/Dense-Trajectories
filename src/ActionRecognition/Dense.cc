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

    //we need to denormalize the images before calculatng the optical flow otherwise the output of it is wrong and scaled
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

	}
}

//create Track objects from a vector of points corresponding to detections in the image
void Dense::makeTracks(std::vector<Track>& tracks, std::vector<cv::Point>&  points, int start_time){
  for (size_t i = 0; i < points.size(); i++) {
    tracks.push_back( Track( points[i], start_time )   );
  }
}

void Dense::track(std::vector<Track>& tracks, Video& flow){

  for (size_t i = 0; i < tracks.size(); i++) {
    if (tracks[i].getLength()<15){

      cv::Point new_point;
      int x=tracks[i].getLastPoint().x;
      int y=tracks[i].getLastPoint().y;
      int t=tracks[i].getLastTime();

      new_point.x = tracks[i].getLastPoint().x + flow[t].at<cv::Point2f>(y,x).x;
      new_point.y = tracks[i].getLastPoint().y + flow[t].at<cv::Point2f>(y,x).y;

      //The tracks sometimes get out of the video for some reason
      if (new_point.x<0 || new_point.x>flow[0].cols){
        continue;
      }
      if (new_point.y<0 || new_point.x>flow[0].rows){
        continue;
      }

      tracks[i].addPoint(new_point);

    }
  }

}

void Dense::extractTrajectories(Video& video, Video& flow, std::vector<Track> & tracks){

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


  // //draw the points on the frame
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


//write tracks to file in a format representable by gnuploy
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


  }

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

void Dense::read_features_from_file_Mat(std::string descriptor_file_path, cv::Mat& features, std::vector<int>& nr_features_per_video){


  std::ifstream desc_file( descriptor_file_path );


  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;
  std::string line;
  getline( desc_file, line );
  std::istringstream buf(line);
  std::istream_iterator<std::string> beg(buf), end;
  std::vector<std::string> tokens(beg, end); // done!

  nr_vectors=atoi(tokens[0].data());
  vector_dimensions=atoi(tokens[1].data());
  nr_videos=atoi(tokens[2].data());

  // std::cout << "nr_vectors" << nr_vectors << '\n';
  // std::cout << "vector_dimensions" << vector_dimensions << '\n';
  // std::cout << "nr_videos" << nr_videos << '\n';

  features=cv::Mat(vector_dimensions,nr_vectors,CV_32FC1);
  MemoryAccessor features_acc (features.rows, features.cols, (Float*) features.data);
  // features.resize(vector_dimensions,nr_vectors);

  int sample=0;
  int features_current_video=0;
  while( getline( desc_file, line ) ){

    if (line=="#"){
      std::cout << "read a video with nr of features"  << features_current_video<< '\n';
      nr_features_per_video.push_back(features_current_video);
      features_current_video=0;
      continue;
    }



    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end); // done!

    for (size_t i = 0; i < tokens.size(); i++) {
      // features.at(i,sample)=atof(tokens[i].data());
      // features_acc(i,sample)=atof(tokens[i].data());
      features.at<float>(i,sample)=atof(tokens[i].data());

    }

    sample++;
    features_current_video++;
  }
  desc_file.close();

}

void Dense::read_features_per_video_from_file_Mat(std::string descriptor_file_path, std::vector<cv::Mat >& features_per_video, int max_nr_videos){

  std::ifstream desc_file( descriptor_file_path );

  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;
  std::string line;
  getline( desc_file, line );
  std::istringstream buf(line);
  std::istream_iterator<std::string> beg(buf), end;
  std::vector<std::string> tokens(beg, end); // done!

  nr_vectors=atoi(tokens[0].data());
  vector_dimensions=atoi(tokens[1].data());
  nr_videos=atoi(tokens[2].data());

  std::cout << "nr_vectors" << nr_vectors << '\n';
  std::cout << "vector_dimensions" << vector_dimensions << '\n';
  std::cout << "nr_videos" << nr_videos << '\n';

  features_per_video.resize(nr_videos);
  for (size_t i = 0; i < nr_videos; i++) {
    features_per_video[i]=cv::Mat(1,1,CV_32FC1);
  }


  std::vector<std::vector<float>>features_video;

  // int nr_features_in_video=0;
  int video_nr=0;
  while( getline( desc_file, line ) ){

    if (line=="#"){
      if (!features_video.empty()){
        // features_per_video[video_nr].resize(vector_dimensions,features_video.size());
        cv::resize(features_per_video[video_nr],features_per_video[video_nr], cv::Size(features_video.size(), vector_dimensions ));
        // cv::resize(features_per_video[video_nr],features_per_video[video_nr], size);


        //get the features_video and put them into the features__video_vector of math::matrices
        for (size_t i = 0; i < vector_dimensions; i++) {
          for (size_t j = 0; j < features_video.size(); j++) {
            // features_per_video[video_nr].at(i,j) = features_video[j][i];
            features_per_video[video_nr].at<float>(i,j) = features_video[j][i];
          }
        }

        features_video.clear();
        video_nr++;
      }
    }

    if (video_nr>max_nr_videos){
      break;
    }


    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end); // done!

    std::vector<float> tmp;
    for (size_t i = 0; i < tokens.size(); i++) {
      tmp.push_back(atof(tokens[i].data()));
    }
    features_video.push_back(tmp);
  }
  desc_file.close();
}

void Dense::read_features_per_video_from_file_math(std::string descriptor_file_path, std::vector<Math::Matrix<Float> >& features_per_video, int max_nr_videos){
  std::ifstream desc_file( descriptor_file_path );


  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;
  std::string line;
  getline( desc_file, line );
  std::istringstream buf(line);
  std::istream_iterator<std::string> beg(buf), end;
  std::vector<std::string> tokens(beg, end); // done!

  nr_vectors=atoi(tokens[0].data());
  vector_dimensions=atoi(tokens[1].data());
  nr_videos=atoi(tokens[2].data());

  std::cout << "nr_vectors" << nr_vectors << '\n';
  std::cout << "vector_dimensions" << vector_dimensions << '\n';
  std::cout << "nr_videos" << nr_videos << '\n';

  features_per_video.resize(nr_videos);

  std::vector<std::vector<float>>features_video;

  // int nr_features_in_video=0;
  int video_nr=0;
  while( getline( desc_file, line ) ){

    if (line=="#"){
      if (!features_video.empty()){
        features_per_video[video_nr].resize(vector_dimensions,features_video.size());

        //get the features_video and put them into the features__video_vector of math::matrices
        for (size_t i = 0; i < vector_dimensions; i++) {
          for (size_t j = 0; j < features_video.size(); j++) {
            features_per_video[video_nr].at(i,j) = features_video[j][i];
          }
        }

        features_video.clear();
        video_nr++;
      }
    }

    if (video_nr>max_nr_videos){
      break;
    }


    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end); // done!

    std::vector<float> tmp;
    for (size_t i = 0; i < tokens.size(); i++) {
      tmp.push_back(atof(tokens[i].data()));
    }
    features_video.push_back(tmp);
  }
  desc_file.close();
}

void Dense::read_features_from_file_math(std::string descriptor_file_path, Math::Matrix<Float>& features){


  std::ifstream desc_file( descriptor_file_path );


  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;
  std::string line;
  getline( desc_file, line );
  std::istringstream buf(line);
  std::istream_iterator<std::string> beg(buf), end;
  std::vector<std::string> tokens(beg, end); // done!

  nr_vectors=atoi(tokens[0].data());
  vector_dimensions=atoi(tokens[1].data());
  nr_videos=atoi(tokens[2].data());

  // std::cout << "nr_vectors" << nr_vectors << '\n';
  // std::cout << "vector_dimensions" << vector_dimensions << '\n';
  // std::cout << "nr_videos" << nr_videos << '\n';

  features.resize(vector_dimensions,nr_vectors);

  int sample=0;
  int features_current_video=0;
  while( getline( desc_file, line ) ){

    if (line=="#"){
      std::cout << "read a video with nr of features"  << features_current_video<< '\n';
      features_current_video=0;
      continue;
    }


    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end); // done!

    for (size_t i = 0; i < tokens.size(); i++) {
      features.at(i,sample)=atof(tokens[i].data());

    }

    sample++;
    features_current_video++;
  }
  desc_file.close();

}

cv::PCA Dense::compressPCA(const cv::Mat& pcaset, int maxComponents, const cv::Mat& testset, cv::Mat& compressed) {
    cv::PCA pca(pcaset, // pass the data
            cv::Mat(), // we do not have a pre-computed mean vector,
                   // so let the PCA engine to compute it
            CV_PCA_DATA_AS_COL, // indicate that the vectors
                                // are stored as matrix rows
                                // (use CV_PCA_DATA_AS_COL if the vectors are
                                // the matrix columns)
            maxComponents // specify, how many principal components to retain
            );
    // if there is no test data, just return the computed basis, ready-to-use
    // if( !testset.data )
    //     return pca;
    // CV_Assert( testset.cols == pcaset.cols );
    compressed.create(maxComponents, pcaset.cols, pcaset.type());
    // cv::Mat reconstructed;
    for( int i = 0; i < pcaset.cols; i++ )
    {
        cv::Mat vec = pcaset.col(i), coeffs = compressed.col(i), reconstructed;
        // compress the vector, the result will be stored
        // in the i-th row of the output matrix
        pca.project(vec, coeffs);
        // and then reconstruct it
        pca.backProject(coeffs, reconstructed);
        // and measure the error
        printf("%d. diff = %g\n", i, norm(vec, reconstructed, cv::NORM_L2));
    }
    return pca;
}

void Dense::write_nr_features_per_video_to_file(std::string nr_features_per_video_file_path, std::vector<int> nr_features_per_video){

  std::ofstream file;
  file.open (nr_features_per_video_file_path);

  for (size_t i = 0; i < nr_features_per_video.size(); i++) {
      file << nr_features_per_video[i] << std::endl;
  }

}

void Dense::write_compressed_features_to_file(std::string desc_compressed_file_path, cv::Mat feat_compressed, std::vector<int> nr_features_per_video){

  std::ofstream file;
  file.open (desc_compressed_file_path);

  int current_video=0;

  int features_written=0;
  for (size_t i = 0; i < feat_compressed.cols; i++) {

      //write the column
      for (size_t j = 0; j < feat_compressed.rows; j++) {
        file << feat_compressed.at<float>(j,i) << " ";
      }
      file << std::endl;
      // file << feat_compressed.col(i) << std::endl;
      features_written++;

      if (features_written==nr_features_per_video[current_video]){
        current_video++;
        features_written=0;
        file << "#"<< std::endl;
      }
  }

  file.seekp(0); //Move at start of file
  file << feat_compressed.cols << " " << 64 << " " << nr_features_per_video.size() << std::endl;
  file.close();

}


void Dense::computeFisherVectors(Math::Matrix<Float>& fisher_vectors, std::vector <Math::Matrix<Float> >&  features_per_video, const Math::Matrix<Float>& means, const Math::Vector<Float>& weights, const Math::Matrix<Float>& sigmas){

  //fisher vectors will contain the vectors for all the videos as column vectors
  fisher_vectors.resize((2*64+1)*64, features_per_video.size() );

  //make a matrix computing the u response of each features in the ideo under each gaussian,  u_k(x_t)
  //rows will be the features, columns will be the gaussians
  for (size_t vid_idx = 0; vid_idx < features_per_video.size(); vid_idx++) {
    std::cout << "calculated fisher vector for video  = " << vid_idx << '\n';
    Math::Matrix<Float> u_s (features_per_video[vid_idx].nColumns(), 64);

    //loop through all of the features and through all of the gaussians and fill the matrix up
    for (size_t x_idx = 0; x_idx < features_per_video[vid_idx].nColumns(); x_idx++) {
      for (size_t k = 0; k < 64; k++) {
        Math::Vector<Float> feature(64);
        features_per_video[vid_idx].getColumn(x_idx, feature);
        u_s.at(x_idx,k)=u(feature, means,sigmas, k);
      }
    }

    //make a vector of the sum of the rows of the u_s matrix, we will need it for the normalization factor in the gammas
    Math::Vector<Float> u_s_sum(features_per_video[vid_idx].nColumns());
    for (size_t x_idx = 0; x_idx < features_per_video[vid_idx].nColumns(); x_idx++) {
      Math::Vector<Float> u_row(64);
      u_s.getRow(x_idx, u_row);
      float res=0;
      for (size_t k = 0; k < 64; k++) {
        res+= weights.at(k)*u_row.at(k);
      }
      u_s_sum.at(x_idx)=res;

    }



    //Make a matrix of gammas containint all the gammas of the featurs in this video
    Math::Matrix<Float> gammas (features_per_video[vid_idx].nColumns(), 64);
    for (size_t x_idx = 0; x_idx < features_per_video[vid_idx].nColumns(); x_idx++) {
      for (size_t k = 0; k < 64; k++) {
        gammas.at(x_idx,k) = (weights.at(k) * u_s.at(x_idx,k)) / u_s_sum.at(x_idx);
      }
    }



    //loop through all of the k in gaussan and get the G_a, G_u and G_sigma
    Math::Vector<Float> fisher_vector((2*64+1)*64);
    for (size_t k = 0; k < 64; k++) {

      //normalization factor
      float normalization=(1/std::sqrt(weights.at(k))) ;


      //sum on the right hand side of G_a
      float sum_g_a=0;
      for (size_t x_idx = 0; x_idx < features_per_video[vid_idx].nColumns(); x_idx++) {
        sum_g_a+= gammas.at(x_idx,k)- weights.at(k);
      }
      float G_a=normalization   * sum_g_a;


      //G_u
      Math::Vector<Float> G_u(64); G_u.setToZero();
      for (size_t x_idx = 0; x_idx < features_per_video[vid_idx].nColumns(); x_idx++) {
        Math::Vector<Float> x(64);
        features_per_video[vid_idx].getColumn(x_idx, x);

        Math::Vector<Float> mean(means.nRows());
        means.getColumn(k, mean);

        Math::Vector<Float> sigma(sigmas.nRows());
    		sigmas.getColumn(k, sigma);
        Math::Vector<Float> sigmaInv(64);
        calculateInverse(sigma, sigmaInv);


        x.add(mean, -1.0f);
        x.elementwiseMultiplication(sigmaInv); //TODO I am not sure what this sigma exactly means

        for (size_t idx = 0; idx < x.nRows(); idx++) {
          x.at(idx)=x.at(idx)*gammas.at(x_idx,k);
        }
        // x.elementwiseMultiplication(gammas.at(x_idx,k));


        G_u.add(x);
      }
      for (size_t idx = 0; idx < G_u.nRows(); idx++) {
        G_u.at(idx)=G_u.at(idx)*normalization;
      }
      // G_u.elementwiseMultiplication(normalization);




      //G_sigma
      Math::Vector<Float> G_s(64); G_s.setToZero();
      for (size_t x_idx = 0; x_idx < features_per_video[vid_idx].nColumns(); x_idx++) {
        Math::Vector<Float> x(64);
        features_per_video[vid_idx].getColumn(x_idx, x);

        Math::Vector<Float> mean(means.nRows());
        means.getColumn(k, mean);

        Math::Vector<Float> sigma(sigmas.nRows());
    		sigmas.getColumn(k, sigma);
        sigma.elementwiseMultiplication(sigma);


        x.add(mean, -1.0f);
        x.elementwiseMultiplication(x);
        x.elementwiseDivision(sigma); //TODO I am not sure what this sigma exactly means
        x.addConstantElementwise(-1);

        for (size_t idx = 0; idx < x.nRows(); idx++) {
          x.at(idx)=x.at(idx)*gammas.at(x_idx,k)* (1/std::sqrt(2));
        }
        // x.elementwiseMultiplication(gammas.at(x_idx,k)* (1/std::sqrt(2)) );


        G_s.add(x);
      }
      for (size_t idx = 0; idx < G_s.nRows(); idx++) {
        G_s.at(idx)=G_s.at(idx)*normalization;
      }
      // G_s.elementwiseMultiplication(normalization);


      //copy them into the respective place in the fisher vector
      fisher_vector.at(k*(2*64+1))=G_a;
      for (size_t i = 0; i < 64; i++) {
        fisher_vector.at(k*(2*64+1) + i + 1)=G_u.at(i);
        fisher_vector.at(k*(2*64+1) + i + k +1 )=G_s.at(i);
      }

    }


    //copy the fisher vector for this video into the big fisher vectors matrix, where each column is the fisher vector for a certain video
    fisher_vectors.setColumn(vid_idx,fisher_vector);




  }


}



Float Dense::calculateDeterminenet(const Math::Vector<Float>& sigma) {
	Float result = 1.0f;
	for (u32 i=0; i<sigma.nRows(); i++) {
		result *= sigma.at(i);
	}
	return result;
}

void Dense::calculateInverse(const Math::Vector<Float>& diagonalMat, Math::Vector<Float>& result) {
	for (u32 i=0; i<diagonalMat.nRows(); i++) {
		result.at(i) = 1.0f / diagonalMat.at(i);
	}
}



float Dense::u(Math::Vector<Float>& x, const Math::Matrix<Float>& means, const Math::Matrix<Float>& sigmas, int k  ){

  Math::Vector<Float> sigma(sigmas.nRows());
  sigmas.getColumn(k, sigma);
  Math::Vector<Float> mean(means.nRows());
  means.getColumn(k, mean);
  Math::Vector<Float> sigmaInv(sigmas.nRows());
  calculateInverse(sigma, sigmaInv);



  x.add(mean, -1.0f);

  Math::Vector<Float> temp(x.nRows());
  temp.copy(x);

  x.elementwiseMultiplication(sigmaInv);
  float result  =  (1.0f/(pow((2 * M_PI), 64 / 2.0f) * pow(abs(calculateDeterminenet(sigma)), 0.5))) * exp(x.dot(temp) * (-1.0f/2.0f));

  return result;

}


void Dense::task_1_2_extract_trajectories(std::string descriptor_file_path){

  Core::AsciiStream in(videoList_, std::ios::in);
  std::string filename;

  std::ofstream desc_file;
  desc_file.open (descriptor_file_path);

  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;


  while (in.getline(filename)) {
    std::cout << "video  " << filename<< '\n';

      Video video;
      Video flow, flowAngle, flowMag;
      Video Lx,Ly,Lt;
      std::vector<Track> tracks;
      readVideo(filename, video);
      // showVideo(video);

      //compute_optical_flow
      opticalFlow(video, flow, flowAngle, flowMag);
      extractTrajectories(video, flow, tracks);
      filterTracks(tracks);

      // std::string trackFile="./tracks.txt";
      // writeTracksToFile(trackFile,tracks);

      //derivatives
      derivatives(video,Lx,Ly,Lt);
      computeDescriptors(video,tracks, Lx, Ly, flow, flowAngle, flowMag);


      //write to file
      int descriptor_written=write_descriptors_to_file(tracks, desc_file);
      desc_file << "#"<< std::endl;

      nr_vectors+=descriptor_written;
      vector_dimensions=426;  //TODO remove hardcode
      nr_videos++;


      video.clear();
      flow.clear();
      flowAngle.clear();
      flowMag.clear();

      Lx.clear();
      Ly.clear();
      Lt.clear();
      tracks.clear();

      std::cout << "finished" << '\n';
  }

  //Add header to file
  desc_file.seekp(0); //Move at start of file
  desc_file << nr_vectors << " " << vector_dimensions << " " << nr_videos << std::endl;
  desc_file.close();

}


void Dense::task_3_pca(std::string descriptor_file_path, std::string desc_compressed_file_path){
  //pca--------------------------------------------------
  //Read the features back again and do pca on them
  std::string nr_features_per_video_file_path = "./new_features_per_video.txt";
  cv::Mat features, feat_compressed;
  std::vector<int> nr_features_per_video;
  read_features_from_file_Mat(descriptor_file_path,features,nr_features_per_video);
  // write_nr_features_per_video_to_file(nr_features_per_video_file_path, nr_features_per_video);

  compressPCA(features, 64, cv::Mat(), feat_compressed);
  std::cout << "finished compressing" << '\n';
  std::cout << "feat compressed has size " << feat_compressed.rows << " " << feat_compressed.cols << '\n';

  write_compressed_features_to_file(desc_compressed_file_path, feat_compressed, nr_features_per_video);

}

void Dense::task_3_gmm(std::string desc_compressed_file_path){

  //Gmm-----------------------------------------
  Math::Matrix<Float> features;
  features.read(desc_compressed_file_path,true);
  // read_features_from_file_math(desc_compressed_file_path,features);
  for (size_t i = 0; i < features.nRows(); i++) {
    for (size_t j = 0; j < features.nColumns(); j++) {
      if (std::isnan( features.at(i,j))  || std::isinf( features.at(i,j))  ){
        std::cout << "error, found a nan in the features" << '\n';
        exit(1);
      }
    }
  }
  std::cout << "compressed features has size" << features.nRows() << " " << features.nColumns()  << '\n';
  Gmm gmm;
  gmm.train(features);
  gmm.save();

}

void Dense::task_3_fisher(std::string desc_compressed_file_path){

  Gmm gmm;
  gmm.load();
  std::cout << "finished loading gmm" << '\n';

  std::vector <Math::Matrix<Float> > features_per_video;
  Math::Matrix<Float> fisher_vectors;
  read_features_per_video_from_file_math(desc_compressed_file_path, features_per_video, 227);

  computeFisherVectors(fisher_vectors, features_per_video, gmm.mean(), gmm.weights(), gmm.sigma());

}





void Dense::run() {
    if (videoList_.empty())
        Core::Error::msg("dense.video-list must not be empty.") << Core::Error::abort;

    // std::string descriptor_file_path = "./desc.txt";
    // std::string desc_compressed_file_path = "./desc_comp.txt";

    std::string descriptor_file_path = "./desc_test.txt";
    std::string desc_compressed_file_path = "./desc_comp_test.txt";

    task_1_2_extract_trajectories(descriptor_file_path);
    task_3_pca(descriptor_file_path, desc_compressed_file_path);
    task_3_gmm(desc_compressed_file_path);
    task_3_fisher(desc_compressed_file_path);
    // task_4_svm(); // TODO


}
