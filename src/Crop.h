//
// Created by Ice on 25-1-21.
//

#ifndef CROP_H
#define CROP_H

#include <opencv2/opencv.hpp>

namespace Algo
{
    cv::Mat Crop(const cv::Mat& image);
    cv::Mat CropBorder(const cv::Mat& image, float rate);
}


#endif //CROP_H
