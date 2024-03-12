#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

int main()
{
  cv::Mat image;
  image = cv::imread("../images/lena.png");
  if (image.empty())
  {
    return 1;
  }
  cv::imshow("lena", image);
  cv::waitKey(0);
  cv::Mat image2;
  cv::dft(image, image2);
  if (image2.empty())
  {
    return 1;
  }
  cv::imshow("image2 after dft", image2);
  cv::waitKey(0);
  return 0;
}
