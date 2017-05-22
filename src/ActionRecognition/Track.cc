#include "Track.hh"

Track::Track(cv::Point point, int start_time){
  m_path.push_back(point);
  m_length=0;
  m_start_time=start_time;

  // std::cout << "made new track at with last time " << m_start_time+m_length << '\n';
}


void Track::addPoint(cv::Point point){
  m_path.push_back(point);
  m_length++;
}

int Track::getLength(){
  return m_length;
}

cv::Point Track::getLastPoint(){
  return m_path[m_length];
}

int Track::getLastTime(){
  return m_start_time+m_length;
}


cv::Point Track::getPoint(int t){
  return m_path[t];
}

int Track::getStartTime(){
  return m_start_time;
}

std::vector<float> Track::getDescriptor(){

  std::vector<float> desc;
  for (size_t i = 0; i < m_path.size()-1; i++) {
    desc.push_back(m_path[i+1].x- m_path[i].x);
    desc.push_back(m_path[i+1].y- m_path[i].y);
  }

  //get norm
  float norm=0;
  for (size_t i = 0; i < desc.size()-2; i=i+2) {
    float disp_norm=0;
    disp_norm+=std::fabs(desc[i])*std::fabs(desc[i]);
    disp_norm+=std::fabs(desc[i+1])*std::fabs(desc[i+1]);
    disp_norm=std::sqrt(disp_norm);
    norm+=disp_norm;
  }

  //normalize
  if (!norm==0.0){
    for (size_t i = 0; i < desc.size(); i++) {
      desc[i]=desc[i]/norm;
    }
  }



  return desc;
}
