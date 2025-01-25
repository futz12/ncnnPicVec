//
// Created by Ice on 25-1-21.
//

#include "MobileCLIPViT.h"

namespace Algo
{
    cv::Mat ExtractKeyFeatures(const cv::Mat& features, int k)
    {
        // 利用SVD提取关键特征
        cv::Mat U, S, V;
        cv::SVD::compute(features, S, U, V);

        cv::Mat result(k, features.cols, features.type());

        for (int i = 0; i < k; i++)
        {
            cv::Mat row = V.row(i) * U.at<float>(i, i);
            // 标准化
            cv::normalize(row, row);
            row.copyTo(result.row(i));
        }

        return result;
    }

    FastViT::FastViT(const std::string& param_path, const std::string& model_path, bool use_gpu)
    {
        model.opt.use_vulkan_compute = use_gpu;

        model.load_param(param_path.c_str());
        model.load_model(model_path.c_str());
    }


    cv::Mat FastViT::forward(const cv::Mat& image)
    {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                     image.cols, image.rows, 224, 224);

        float mean_vals[3] = {0, 0, 0};
        float norm_vals[3] = {1.0f / 255, 1.0f / 255, 1.0f / 255};
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto ex = model.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out0;
        ex.extract("out0", out0);

        cv::Mat result(out0.h, out0.w, CV_32F);
        memcpy(result.data, out0.data, out0.h * out0.w * sizeof(float));
        // 标准化
        cv::normalize(result, result);
        return result;
    }
} // Algo
