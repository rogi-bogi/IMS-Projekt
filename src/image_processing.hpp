#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// source: https://learnopencv.com/opencv-threshold-python-cpp/
// source: https://learnopencv.com/image-filtering-using-convolution-in-opencv/

cv::Mat getDftOfBWImage(cv::Mat src) // source: https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
{
  cv::Mat paddedImage;
  uint16_t srcRows = src.rows;
  uint16_t srcCols = src.cols;
  uint16_t optimalRows = cv::getOptimalDFTSize(srcRows);
  uint16_t optimalCols = cv::getOptimalDFTSize(srcCols);
  cv::copyMakeBorder(src, paddedImage, 0, optimalRows - srcRows, 0, optimalCols - srcCols, cv::BORDER_CONSTANT,
                     cv::Scalar::all(0));

  cv::Mat planes[] = {cv::Mat_<float>(paddedImage), cv::Mat::zeros(paddedImage.size(), CV_32F)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI); // Add to the expanded another plane with zeros

  cv::dft(complexI, complexI); // this way the result may fit in the source matrix

  cv::split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
  cv::Mat magI = planes[0];

  magI += cv::Scalar::all(1); // switch to logarithmic scale
  cv::log(magI, magI);

  // crop the spectrum, if it has an odd number of rows or columns
  magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

  // rearrange the quadrants of Fourier image so that the origin is at the image center
  int cx = magI.cols / 2;
  int cy = magI.rows / 2;

  cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

  cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  cv::normalize(magI, magI, 0, 1,
                cv::NORM_MINMAX); // Transform the matrix with float values into a viewable image form (float between
                                  // values 0 and 1).

  return magI;
}

// Function to apply a frequency filter to a 2D complex matrix
void applyFilter(vector<vector<Complex>>& data, int rows, int cols, function<bool(double, double)> filterFunction) {
  // Loop through each element in the frequency domain (data)
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      // Calculate normalized frequencies (between -0.5 and 0.5)
      double u = (static_cast<double>(j) / (cols - 1)) - 0.5;
      double v = (static_cast<double>(i) / (rows - 1)) - 0.5;

      // Apply the filter function to get filter value (0 for filtering, 1 for keeping)
      double filterVal = filterFunction(u, v);

      // Modify the complex element based on the filter value
      data[i][j] *= filterVal;
    }
  }
}

// Define filter function (replace with your desired filter characteristics)
double ourFilter(double u, double v) {
  double cutoffFrequency = 50.0;  // Adjust this value based on your needs
  double distance = sqrt(u * u + v * v);
  return (distance <= cutoffFrequency) ? 1.0 : 0.0;
}

void addNoise(cv::Mat src)
{
  if (src.empty())
  {
    return;
  }
  cv::Mat noise(src.size(), src.type());
  constexpr auto mean = 5u;
  constexpr auto var = 20u;
  cv::randn(noise, mean, var);
  src += noise;
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

