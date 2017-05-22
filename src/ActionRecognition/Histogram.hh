#ifndef HISTOGRAM_HH_
#define HISTOGRAM_HH_

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iterator>


class Histogram
{
public:
  Histogram();
  Histogram(int nbins, float range);
  void init(int nbins, float range);
  void add_val(float val, float weight);
  void normalize();
  void concatenate(Histogram& hist);
  void concatenate(std::vector<float>& hist);
  int size();
  // std::vector<float> descriptor();
  std::string to_string();

private:
  float m_range;
  float m_bin_size;
  std::vector<float> m_hist;

};


#endif
