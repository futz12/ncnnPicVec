//
// Created by Ice on 25-1-21.
//

#include "Crop.h"

cv::Mat Algo::Crop(const cv::Mat& image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat thresh;
    // 拉普拉斯算子相对敏感，因此可能裁出边框来
    cv::Laplacian(gray, thresh, CV_8U, 3);
    // 膨胀，最大化边缘
    cv::dilate(thresh, thresh, cv::Mat());

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return {};
    }

    int size_y = image.rows;

    std::vector<cv::Rect> bounds;
    for (const auto& contour : contours)
    {
        bounds.push_back(cv::boundingRect(contour));
    }


    auto [x0,y0,w0,h0] = bounds[0];

    for (auto [x,y,w,h] : bounds)
    {
        if (w * h * (size_y / 2  - abs(size_y / 2 - y - h / 2)) > w0 * h0 * (size_y / 2 - abs(size_y / 2 - y0 - h0 / 2))) // 优先中间部分的图片
        {
            x0 = x;
            y0 = y;
            w0 = w;
            h0 = h;
        }
    }

    cv::Mat result = image(cv::Rect(x0, y0, w0, h0));

    return result;
}

cv::Mat Algo::CropBorder(const cv::Mat& image, float rate)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat thresh;

    cv::Canny(gray, thresh, 120, 200);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return image.clone();
    }

    std::vector<cv::Rect> bounds;
    for (const auto& contour : contours)
    {
        bounds.push_back(cv::boundingRect(contour));
    }


    auto [x0,y0,w0,h0] = bounds[0];

    for (auto [x,y,w,h] : bounds)
    {
        if (w * h > w0 * h0) // 优先中间部分的图片
        {
            x0 = x;
            y0 = y;
            w0 = w;
            h0 = h;
        }
    }

    if (w0 * h0 > rate * image.rows * image.cols)
    {
        cv::Mat result = image(cv::Rect(x0, y0, w0, h0));
        return result;
    }
    else
    {
        return image.clone();
    }
}
