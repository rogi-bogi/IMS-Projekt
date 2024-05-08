#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "helpers.hpp"

using namespace cv;

void calculateDFT(Mat& scr, Mat& dst)
{
  // define mat consists of two mat, one for real values and the other for complex values
  Mat planes[] = {scr, Mat::zeros(scr.size(), CV_32F)};
  Mat complexImg;
  merge(planes, 2, complexImg);
  dft(complexImg, complexImg);
  dst = complexImg;
}

// IDFT
Mat reverseDTF(Mat filteredFD)
{
  Mat imgOut;
  dft(filteredFD, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);
  normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);
  return imgOut;
}

void fftshift(const Mat& input_img, Mat& output_img)
{
  output_img = input_img.clone();
  int cx = output_img.cols / 2;
  int cy = output_img.rows / 2;
  Mat q1(output_img, Rect(0, 0, cx, cy));
  Mat q2(output_img, Rect(cx, 0, cx, cy));
  Mat q3(output_img, Rect(0, cy, cx, cy));
  Mat q4(output_img, Rect(cx, cy, cx, cy));

  Mat temp;
  q1.copyTo(temp);
  q4.copyTo(q1);
  temp.copyTo(q4);
  q2.copyTo(temp);
  q3.copyTo(q2);
  temp.copyTo(q3);
}

// Frequency domain filter matrix as "H" (common in literature)
Mat construct_H(Mat& scr, String type, float D0)
{
  // Matrix filled with 1's (all-pass filter)
  Mat H(scr.size(), CV_32F, Scalar(1));
  float D = 0;
  if (type == "Ideal LP")
  {
    for (int u = 0; u < H.rows; u++)
    {
      for (int v = 0; v < H.cols; v++)
      {
        D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
        if (D > D0)
        {
          H.at<float>(u, v) = 0;
        }
      }
    }
  }
  else if (type == "Gaussian LP")
  {
    for (int u = 0; u < H.rows; u++)
    {
      for (int v = 0; v < H.cols; v++)
      {
        D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
        H.at<float>(u, v) = exp(-D * D / (2 * D0 * D0));
      }
    }
  }
  else if (type == "Ideal HP")
  {
    for (int u = 0; u < H.rows; u++)
    {
      for (int v = 0; v < H.cols; v++)
      {
        D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
        if (D < D0)
        {
          H.at<float>(u, v) = 0;
        }
      }
    }
  }
  else if (type == "Gaussian HP")
  {
    for (int u = 0; u < H.rows; u++)
    {
      for (int v = 0; v < H.cols; v++)
      {
        D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
        H.at<float>(u, v) = 1 - exp(-D * D / (2 * D0 * D0));
      }
    }
  }
  else if (type == "BandPass")
  {
    float D1 = D0 * 0.75;
    float D2 = D0 * 1.25;
    for (int u = 0; u < H.rows; u++)
    {
      for (int v = 0; v < H.cols; v++)
      {
        D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
        if (D < D1 || D > D2)
        {
          H.at<float>(u, v) = 0;
        }
      }
    }
  }
  else if (type == "Notch")
  {
    float D1 = D0 * 0.75;
    float D2 = D0 * 1.25;
    for (int u = 0; u < H.rows; u++)
    {
      for (int v = 0; v < H.cols; v++)
      {
        D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
        if (D >= D1 && D <= D2)
        {
          H.at<float>(u, v) = 0;
        }
      }
    }
  }
  return H;
}

void filtering(Mat& scr, Mat& dst, Mat& H)
{
  fftshift(H, H);
  Mat planesH[] = {Mat_<float>(H.clone()), Mat_<float>(H.clone())};

  Mat planes_dft[] = {scr, Mat::zeros(scr.size(), CV_32F)};
  split(scr, planes_dft);

  Mat planes_out[] = {Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F)};
  planes_out[0] = planesH[0].mul(planes_dft[0]);
  planes_out[1] = planesH[1].mul(planes_dft[1]);

  merge(planes_out, 2, dst);
}

void show_dft_effect(Mat image)
{
  // Expanding input image to optimal size
  Mat padded;
  int m = getOptimalDFTSize(image.rows);
  int n = getOptimalDFTSize(image.cols);
  copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
  /*
    The result of the transformation is complex numbers.
    Displaying this is possible via a magnitude.
        */
  Mat real, imaginary;
  Mat planes[] = {real, imaginary};

  split(padded, planes);
  Mat mag_image;
  magnitude(planes[0], planes[1], mag_image);

  // Switch to a logarithmic scale
  mag_image += Scalar::all(1);
  log(mag_image, mag_image);
  mag_image = mag_image(Rect(0, 0, mag_image.cols & -2, mag_image.rows & -2));

  Mat shifted_DFT;
  fftshift(mag_image, shifted_DFT);

  normalize(shifted_DFT, shifted_DFT, 0, 1, NORM_MINMAX);

  imshow("After DFT", shifted_DFT);
  waitKey(0);
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
int apply_build_in_functions(Mat& imgIn)
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
  displayImages(filteredImages, imWindowNames);

  return 0;
}
