#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Function to display an image
void displayImage(const cv::Mat& image, const std::string& windowName) {
    cv::imshow(windowName, image);
    cv::waitKey();
    cv::destroyAllWindows();
}

// Function to save an image
void saveImage(const cv::Mat& image, const std::string& filename) {
    cv::imwrite(filename, image);
}

// Function to display multiple images
void displayImages(const std::vector<cv::Mat>& images, const std::vector<std::string>& windowNames) {
    for (size_t i = 0; i < images.size(); ++i) {
        cv::namedWindow(windowNames[i], cv::WINDOW_NORMAL);
        cv::imshow(windowNames[i], images[i]);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}