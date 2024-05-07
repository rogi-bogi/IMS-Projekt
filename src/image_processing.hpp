#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

// // Function to apply a frequency filter to a 2D complex matrix
// void applyFilter(vector<vector<Complex>>& data, int rows, int cols, function<bool(double, double)> filterFunction) {
//   // Loop through each element in the frequency domain (data)
//   for (int i = 0; i < rows; ++i) {
//     for (int j = 0; j < cols; ++j) {
//       // Calculate normalized frequencies (between -0.5 and 0.5)
//       double u = (static_cast<double>(j) / (cols - 1)) - 0.5;
//       double v = (static_cast<double>(i) / (rows - 1)) - 0.5;

//       // Apply the filter function to get filter value (0 for filtering, 1 for keeping)
//       double filterVal = filterFunction(u, v);

//       // Modify the complex element based on the filter value
//       data[i][j] *= filterVal;
//     }
//   }
// }

////////////////////////////////////////////////////////////////////////////
////////////////////              BUILT-INS             ////////////////////
////////////////////////////////////////////////////////////////////////////

// Kernels are in previous code

// Define filter function (replace with your desired filter characteristics)
double ourFilter(double u, double v) {
  double cutoffFrequency = 50.0;  // Adjust this value based on your needs
  double distance = sqrt(u * u + v * v);
  return (distance <= cutoffFrequency) ? 1.0 : 0.0;
}

// Function to apply identity filter using a specified kernel
cv::Mat applyIdentityFilter(const cv::Mat& inputImage, const cv::Mat& kernel) {
    cv::Mat filteredImage;
    cv::filter2D(inputImage, filteredImage, -1 , kernel, cv::Point(-1, -1), 0, 4);
    return filteredImage;
}

// Function to blur an image using a specified kernel
cv::Mat applyBlurFilter(const cv::Mat& inputImage, const cv::Mat& kernel) {
    cv::Mat filteredImage;
    cv::filter2D(inputImage, filteredImage, -1 , kernel, cv::Point(-1, -1), 0, 4);
    return filteredImage;
}

// Function to apply Gaussian blur to an image
cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize) {
    cv::Mat blurredImage;
    //Gaussian kernel standard deviations
    int sigmaX = 0;
    int sigmaY = 0;
    cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), sigmaX, sigmaY);
    return blurredImage;
}

// Function to apply sharpening using a specified kernel
cv::Mat applySharpening(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat sharpImage;
    cv::filter2D(image, sharpImage, -1 , kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    return sharpImage;
}

