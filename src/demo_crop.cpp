//
// Created by Ice on 25-1-21.
//

#include <iostream>
#include <Crop.h>


int main()
{
    cv::Mat image;
    image = cv::imread("../images/7.jpg");
    cv::imshow("Original", image);
    cv::Mat result = Algo::Crop(image);
    result = Algo::CropBorder(result, 0.9);
    cv::imshow("Cropped", result);
    cv::waitKey(0);
    return 0;
}
