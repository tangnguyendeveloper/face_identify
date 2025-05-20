#ifndef DRAW_HPP
#define DRAW_HPP

#include "mtcnn/face.h"


using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data) {
  cv::Mat outImg;
  img.convertTo(outImg, CV_8UC3);

  for (auto &d : data) {
    cv::rectangle(outImg, d.first, cv::Scalar(0, 255, 255), 2);
    auto pts = d.second;
    for (size_t i = 0; i < pts.size(); ++i) {
      cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
    }
  }
  return outImg;
}


static cv::Mat getDrawFacesImage(const cv::Mat &img,
                                  const std::vector<Face> &faces) {
  std::vector<rectPoints> data;
  for (size_t i = 0; i < faces.size(); ++i) {
    std::vector<cv::Point> pts;
    for (int p = 0; p < NUM_PTS; ++p) {
      pts.push_back(
          cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
    }
    auto rect = faces[i].bbox.getRect();
    auto d = std::make_pair(rect, pts);
    data.push_back(d);
  }
  return drawRectsAndPoints(img, data);
}


#endif // DRAW_HPP