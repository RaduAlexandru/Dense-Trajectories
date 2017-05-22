#pragma once

#include <opencv2/core/core.hpp>
#include <iostream>
#include "Histogram.hh"

class Track{
public:
  Track(cv::Point point, int start_time);
  void addPoint(cv::Point point);
  int getLength();
  cv::Point getLastPoint();
  int getLastTime();
  cv::Point getPoint(int t);
  int getStartTime();
  std::vector<float> getDescriptor();
  Histogram descriptor;


private:
  std::vector<cv::Point> m_path;
  int m_length;
  int m_start_time;

};
