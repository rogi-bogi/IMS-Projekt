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

  cv::Mat outputImage(inputImage);
  addNoise(outputImage);
  cv::imshow("lena after", outputImage);

  cv::waitKey(0);

  return 0;
}
