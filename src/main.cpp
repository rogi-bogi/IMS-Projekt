#include "FFT.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

void addNoise(cv::Mat& src)
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

cv::Mat getDftOfBWImage(cv::Mat src)
{
  // source: https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
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

int main()
{
  cv::Mat inputImage;
  inputImage = cv::imread("../images/lena.png", cv::IMREAD_GRAYSCALE);
  if (inputImage.empty())
  {
    return 1;
  }
  cv::imshow("lena", inputImage);
  cv::waitKey(0);

  cv::Mat outputImage = getDftOfBWImage(inputImage);
  // addNoise(outputImage);
  cv::imshow("lena after", outputImage);

  cv::waitKey(0);

  return 0;
}
