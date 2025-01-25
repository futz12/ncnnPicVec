//
// Created by Ice on 25-1-21.
//

#ifndef MOBILECLIPVIT_H
#define MOBILECLIPVIT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ncnn/mat.h>
#include <ncnn/net.h>

namespace Algo
{
    cv::Mat ExtractKeyFeatures(const cv::Mat& features, int k);

    class FastViT
    {
    public:
        FastViT(const std::string& param_path, const std::string& model_path, bool use_gpu = false);

        cv::Mat forward(const cv::Mat& image);

    private:
        ncnn::Net model;
    };
} // Algo

#endif //MOBILECLIPVIT_H
