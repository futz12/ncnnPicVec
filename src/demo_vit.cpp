//
// Created by Ice on 25-1-21.
//

#include <iostream>
#include <MobileCLIPViT.h>

int main()
{
    cv::Mat image;
    image = cv::imread("../images/10.jpg");
    cv::imshow("Original", image);
    Algo::FastViT model("../models/mobileclip_s0_fp16.param", "../models/mobileclip_s0_fp16.bin");
    auto result = model.forward(image);
    std::cout << result << std::endl;
    cv::waitKey(0);
    return 0;
}