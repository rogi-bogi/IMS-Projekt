#include <iostream>

#include "FFT.hpp"
#include "image_processing.hpp"
#include "helpers.hpp"

//TODO: cv::butterworthFilter or cv::chebyshevFilter

int main()
{
  cv::Mat inputImage;
  inputImage = cv::imread("../images/lena.png", cv::IMREAD_GRAYSCALE);
  if (inputImage.empty())
  {
    cout << "Could not read image" << endl;
    return 1;
  }
  
  // Define kernels
  cv::Mat kernel1 = (cv::Mat_<double>(3,3) << 0, 0, 0, 
                                              0, 1, 0,
                                              0, 0, 0);
  cv::Mat kernel2 = cv::Mat::ones(5,5, CV_64F) / 25;
  int kernel3_size = 3;
  cv::Mat kernel4 = (cv::Mat_<double>(3,3) << 0, -1,  0, 
                                            -1,  5, -1, 
                                             0, -1,  0);

  // Apply identity filter using kernel1
  cv::Mat identityImage = applyIdentityFilter(inputImage, kernel1);

  // Apply blur filter using kernel2
  cv::Mat blurredImage = applyBlurFilter(inputImage, kernel2);

  // Apply Gaussian blur
  cv::Mat gaussianBlurredImage = applyGaussianBlur(inputImage, kernel3_size);

  // Apply sharpening
  cv::Mat sharpenedImage = applySharpening(inputImage, kernel4);

  std::vector<cv::Mat> filteredImages = {inputImage, identityImage, blurredImage, gaussianBlurredImage, sharpenedImage};
  std::vector<std::string> imWindowNames = {"Original", "Identity", "Kernel Blur", "Gaussian Blur", "Sharpening"};

  // Display images
  displayImages(filteredImages, imWindowNames);

  // Comparing frequency domains
  cv::Mat outputImage = getDftOfBWImage(inputImage);
  cv::Mat opIdentity = getDftOfBWImage(identityImage);
  cv::Mat opBlurred = getDftOfBWImage(blurredImage);
  cv::Mat opGaussianBlurred = getDftOfBWImage(gaussianBlurredImage); 
  cv::Mat opSharpened = getDftOfBWImage(sharpenedImage); 

  std::vector<cv::Mat> frequencyDomains = {outputImage, opIdentity, opBlurred, opGaussianBlurred, opSharpened};
  std::vector<std::string> fdWindowNames = {"Frequency Domain", "Identity FD", "Blurred FD", "Gaussian Blur FD", "Sharpening FD"};

  // Display images
  displayImages(frequencyDomains, fdWindowNames);

  return 0;
}
