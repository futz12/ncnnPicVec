//
// Created by Ice on 25-1-21.
//

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <MobileCLIPViT.h>

std::vector<std::string> ListFiles(const std::string& path)
{
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        files.push_back(entry.path().string());
    }
    return files;
}

int main()
{
    const int key_size = 3; // 提取关键特征的数量
    const float threshold = 0.6; // 0 - 2，越小越严格

    Algo::FastViT model("../models/mobileclip_s0_fp16.param", "../models/mobileclip_s0_fp16.bin");
    std::string prefix = "../images/";

    std::vector<std::string> classes = {"GenShin", "CatMeme", "MyGO","Mao","Panda"};

    cv::Mat features;

    for (auto& cls : classes)
    {
        std::string path = prefix + cls;
        auto files = ListFiles(path);
        cv::Mat feature;
        for (auto& file : files)
        {
            cv::Mat image = cv::imread(file);
            auto result = model.forward(image);
            feature.push_back(result);
        }
        features.push_back(Algo::ExtractKeyFeatures(feature, key_size));
    }

    cv::flann::Index index(features, cv::flann::KDTreeIndexParams(4));

    // Val
    std::string val_path = "Val";

    auto files = ListFiles(prefix + val_path);

    for (auto& file : files)
    {
        cv::Mat image = cv::imread(file);
        auto query = model.forward(image);

        cv::Mat indices, dists;
        index.knnSearch(query, indices, dists, 1);
        std::cout << "Query: " << file << std::endl;

        if (dists.at<float>(0) > threshold)
        {
            std::cout << "Result: Unknown" << std::endl;
            std::cout << "Distance: " << dists.at<float>(0) << std::endl;
            std::cout << "==================" << std::endl;
            continue;
        }
        std::cout << "Result: " << classes[indices.at<int>(0) / key_size] << std::endl;
        std::cout << "Distance: " << dists.at<float>(0) << std::endl;
        std::cout << "==================" << std::endl;
    }

    return 0;
}
