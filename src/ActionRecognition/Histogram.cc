#include "Histogram.hh"

Histogram::Histogram(){

}

Histogram::Histogram(int nbins, float range){
  // std::cout << "hist constructor" << '\n';
  m_hist.resize(nbins,0);
  m_range=range;
  m_bin_size=range/nbins;
  // std::cout << "m_bins size is" << m_bin_size << '\n';

}

void Histogram::init(int nbins, float range){
  m_hist.resize(nbins);
  m_range=range;
  m_bin_size=range/nbins;
}


void Histogram::add_val(float val, float weight){

  //each bin has a center located at (bin_size)*i - (bin_size/2) where i is has range 1---nbins
  //Interpolation between bins done according to: http://stackoverflow.com/questions/6565412/hog-trilinear-interpolation-of-histogram-bins

  int closest_bin= val/m_bin_size;


  //decide if it is on the right or left of the center
  float center_closest=(m_bin_size)*(closest_bin+1)  - m_bin_size/2.0;
  int second_closest=-1;
  if (val<center_closest){  //seond closest is to the left
    if (closest_bin==0)
      second_closest=m_hist.size()-1;
    else
      second_closest=closest_bin-1;
  }else{  //second closest is to the right
    if (closest_bin==(m_hist.size()-1))
      second_closest=0;
    else
      second_closest=closest_bin+1;
  }

  m_hist[closest_bin]= m_hist[closest_bin] + weight* (1 - (val- center_closest)/m_bin_size);
  m_hist[second_closest]= m_hist[closest_bin] + weight* ((val- center_closest)/m_bin_size);


}


void Histogram::normalize(){

  //calculate norm
  float norm=0.0;
  for (size_t i = 0; i < m_hist.size(); i++) {
    norm+=  m_hist[i]*m_hist[i];
  }
  norm=std::sqrt(norm);

  //normalize
  if (!norm==0.0){
    for (size_t i = 0; i < m_hist.size(); i++) {
      m_hist[i]=m_hist[i]/norm;
    }
  }

}


void Histogram::concatenate(Histogram& hist){
  m_hist.insert(m_hist.end(),  hist.m_hist.begin(), hist.m_hist.end());
}

void Histogram::concatenate(std::vector<float>& hist){
  m_hist.insert(m_hist.end(),  hist.begin(), hist.end());
}

int Histogram::size(){
  return m_hist.size();
}

// std::vector<float> Histogram::descriptor(){
//   return m_hist;
// }

std::string Histogram::to_string(){
  std::string desc_string;

  std::ostringstream oss;

  if (!m_hist.empty()){
    // Convert all but the last element to avoid a trailing ","
    std::copy(m_hist.begin(), m_hist.end()-1, std::ostream_iterator<float>(oss, " "));

    // Now add the last element with no delimiter
    oss << m_hist.back();
  }

  return oss.str();

}
