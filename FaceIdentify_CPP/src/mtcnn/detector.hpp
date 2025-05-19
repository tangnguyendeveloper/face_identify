#ifndef _include_opencv_mtcnn_detector_h_
#define _include_opencv_mtcnn_detector_h_

#include "face.h"
#include "onet.h"
#include "pnet.h"
#include "rnet.h"
#include "config/models_config.hpp"

class MTCNNDetector {
private:
  std::unique_ptr<ProposalNetwork> _pnet;
  std::unique_ptr<RefineNetwork> _rnet;
  std::unique_ptr<OutputNetwork> _onet;

public:
  MTCNNDetector(const ModelsConfig &config);
  std::vector<Face> detect(const cv::Mat &img, const float minFaceSize,
                           const float scaleFactor);
};

#endif
