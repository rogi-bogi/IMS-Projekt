#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "fft_filters.hpp"
#include "helpers.hpp"

namespace image_processing
{

cv::Mat generateHistogram(const cv::Mat& img)
{
  // Establish the number of bins
  int histSize = 256;

  // Set the range of values
  float range[] = {0, 256};
  const float* histRange = {range};

  cv::Mat hist;
  cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

  return hist;
}

void showHistogram(const cv::Mat& img, std::string plotTitle = "Histogram")
{
  cv::Mat hist = generateHistogram(img);
  // Parameters for the histogram image
  int hist_w = 512;                          // width of the histogram image
  int hist_h = 400;                          // height of the histogram image
  int bin_w = cvRound((double)hist_w / 256); // width of each bin
  cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255, 255, 255));
  cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  // Draw the histogram
  for (int i = 1; i < 256; i++)
  {
    cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), cv::Scalar(0, 0, 0), 2, 8, 0);
  }

  // Draw axes
  cv::line(histImage, cv::Point(0, hist_h - 1), cv::Point(hist_w, hist_h - 1), cv::Scalar(0, 0, 0));
  cv::line(histImage, cv::Point(0, 0), cv::Point(0, hist_h), cv::Scalar(0, 0, 0));

  // Add value markers for the x-axis
  for (int i = 0; i < 256; i += 32)
  {
    cv::line(histImage, cv::Point(bin_w * i, hist_h - 1), cv::Point(bin_w * i, hist_h - 10), cv::Scalar(0, 0, 0));
    cv::putText(histImage, std::to_string(i), cv::Point(bin_w * i, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0, 0, 0));
  }

  // Add value markers for the y-axis
  double minVal, maxVal;
  cv::minMaxLoc(hist, &minVal, &maxVal);
  for (int i = 0; i <= 5; i++)
  {
    int y = hist_h - (i * hist_h / 5);
    cv::line(histImage, cv::Point(0, y), cv::Point(10, y), cv::Scalar(0, 0, 0));
    cv::putText(histImage, std::to_string(cvRound(i * maxVal / 5)), cv::Point(15, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0, 0, 0));
  }

  // Display the histogram image
  cv::imshow(plotTitle, histImage);
}

void calculateDFT(cv::Mat& scr, cv::Mat& dst)
{
  cv::Mat scr32;
  // Converting from 8-bit to float type suitable for DFT and Wavelets operations
  scr.convertTo(scr32, CV_32F);
  // define mat consists of two mat, one for real values and the other for complex values
  cv::Mat planes[] = {scr32, cv::Mat::zeros(scr.size(), CV_32F)};
  cv::Mat complexImg;
  merge(planes, 2, complexImg);
  dft(complexImg, complexImg);
  dst = complexImg;
}

// IDFT
cv::Mat reverseDTF(cv::Mat filteredFD)
{
  cv::Mat imgOut;
  dft(filteredFD, imgOut, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  normalize(imgOut, imgOut, 0, 1, cv::NORM_MINMAX);
  return imgOut;
}

void fftshift(const cv::Mat& input_img, cv::Mat& output_img)
{
  output_img = input_img.clone();
  int cx = output_img.cols / 2;
  int cy = output_img.rows / 2;
  cv::Mat q1(output_img, cv::Rect(0, 0, cx, cy));
  cv::Mat q2(output_img, cv::Rect(cx, 0, cx, cy));
  cv::Mat q3(output_img, cv::Rect(0, cy, cx, cy));
  cv::Mat q4(output_img, cv::Rect(cx, cy, cx, cy));

  cv::Mat temp;
  q1.copyTo(temp);
  q4.copyTo(q1);
  temp.copyTo(q4);
  q2.copyTo(temp);
  q3.copyTo(q2);
  temp.copyTo(q3);
}

// Frequency domain filter matrix as "H" (common in literature)
// Default n and epsilon allow to use function without these values
cv::Mat construct_H(cv::Mat& scr, std::string type, float D0, int n = 0, float epsilon = 0.0f)
{
  // Matrix filled with 1's (all-pass filter)
  cv::Mat H(scr.size(), CV_32F, cv::Scalar(1));
  float D = 0;
  if (type == "Ideal LP")
  {
    idealLpFilter(scr, H, D, D0);
  }
  else if (type == "Gaussian LP")
  {
    gaussianLpFilter(scr, H, D, D0);
  }
  else if (type == "Ideal HP")
  {
    idealHpFilter(scr, H, D, D0);
  }
  else if (type == "Gaussian HP")
  {
    gaussianHpFilter(scr, H, D, D0);
  }
  else if (type == "BandPass")
  {
    bandPassFilter(scr, H, D, D0);
  }
  else if (type == "Notch")
  {
    notchFilter(scr, H, D, D0);
  }
  else if (type == "Butterworth LP")
  {
    butterworthLpFilter(scr, H, D, D0, n);
  }
  else if (type == "Chebyshev LP")
  {
    chebyshevLpFilter(scr, H, D, D0, epsilon, n);
  }
  return H;
}

void filtering(cv::Mat& scr, cv::Mat& dst, cv::Mat& H)
{
  fftshift(H, H);
  cv::Mat planesH[] = {cv::Mat_<float>(H.clone()), cv::Mat_<float>(H.clone())};

  cv::Mat planes_dft[] = {scr, cv::Mat::zeros(scr.size(), CV_32F)};
  split(scr, planes_dft);

  cv::Mat planes_out[] = {cv::Mat::zeros(scr.size(), CV_32F), cv::Mat::zeros(scr.size(), CV_32F)};
  planes_out[0] = planesH[0].mul(planes_dft[0]);
  planes_out[1] = planesH[1].mul(planes_dft[1]);

  merge(planes_out, 2, dst);
}

void show_dft_effect(cv::Mat image)
{
  // Expanding input image to optimal size
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(image.rows);
  int n = cv::getOptimalDFTSize(image.cols);
  copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  /*
    The result of the transformation is complex numbers.
    Displaying this is possible via a magnitude.
        */
  cv::Mat real, imaginary;
  cv::Mat planes[] = {real, imaginary};

  split(padded, planes);
  cv::Mat mag_image;
  magnitude(planes[0], planes[1], mag_image);

  // Switch to a logarithmic scale
  mag_image += cv::Scalar::all(1);
  log(mag_image, mag_image);
  mag_image = mag_image(cv::Rect(0, 0, mag_image.cols & -2, mag_image.rows & -2));

  cv::Mat shifted_DFT;
  fftshift(mag_image, shifted_DFT);

  normalize(shifted_DFT, shifted_DFT, 0, 1, cv::NORM_MINMAX);

  imshow("After DFT", shifted_DFT);
  cv::waitKey(0);
}

////////////////////////////////////////////////////////////////////////////
////////////////////              BUILT-INS             ////////////////////
////////////////////////////////////////////////////////////////////////////

// Function to apply identity filter using a specified kernel
cv::Mat applyIdentityFilter(const cv::Mat& inputImage, const cv::Mat& kernel)
{
  cv::Mat filteredImage;
  cv::filter2D(inputImage, filteredImage, -1, kernel, cv::Point(-1, -1), 0, 4);
  return filteredImage;
}

// Function to blur an image using a specified kernel
cv::Mat applyBlurFilter(const cv::Mat& inputImage, const cv::Mat& kernel)
{
  cv::Mat filteredImage;
  cv::filter2D(inputImage, filteredImage, -1, kernel, cv::Point(-1, -1), 0, 4);
  return filteredImage;
}

// Function to apply Gaussian blur to an image
cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize)
{
  cv::Mat blurredImage;
  // Gaussian kernel standard deviations
  int sigmaX = 0;
  int sigmaY = 0;
  cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), sigmaX, sigmaY);
  return blurredImage;
}

// Function to apply sharpening using a specified kernel
cv::Mat applySharpening(const cv::Mat& image, const cv::Mat& kernel)
{
  cv::Mat sharpImage;
  cv::filter2D(image, sharpImage, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
  return sharpImage;
}

// PREVIOUS BUILT-INS IMPLEMENTATION
int apply_build_in_functions(cv::Mat& imgIn)
{

  // Define kernels
  cv::Mat kernel1 = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
  cv::Mat kernel2 = cv::Mat::ones(5, 5, CV_64F) / 25;
  int kernel3_size = 3;
  cv::Mat kernel4 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

  // Apply identity filter using kernel1
  cv::Mat identityImage = applyIdentityFilter(imgIn, kernel1);

  // Apply blur filter using kernel2
  cv::Mat blurredImage = applyBlurFilter(imgIn, kernel2);

  // Apply Gaussian blur
  cv::Mat gaussianBlurredImage = applyGaussianBlur(imgIn, kernel3_size);

  // Apply sharpening
  cv::Mat sharpenedImage = applySharpening(imgIn, kernel4);

  std::vector<cv::Mat> filteredImages = {imgIn, identityImage, blurredImage, gaussianBlurredImage, sharpenedImage};
  std::vector<std::string> imWindowNames = {"Original", "Identity", "Kernel Blur", "Gaussian Blur", "Sharpening"};

  // Display images
  helpers::displayImages(filteredImages, imWindowNames);

  return 0;
}
} // namespace image_processing
