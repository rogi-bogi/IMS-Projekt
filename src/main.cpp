#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/highgui.hpp>

#include "FFT.hpp"
#include "helpers.hpp"
#include "image_processing.hpp"
#include "wavelets.hpp"

using namespace cv;

/*
        TODO: cv::butterworthFilter or cv::chebyshevFilter ??
        TODO: Make some console program logic:
                        * Image as path argument - for different images
                        * Loop behaviour - choosing filter over and over
                        * Maybe some live preview of changes (depending on frequency parameter)
*/

// For file as an cmd argument
/*
int main(int argc, char** argv) {
  // Check for input argument
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }
  // Read the PNG image
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "Error: Could not read image" << std::endl;
    return 1;
  }
*/

void menuLoop(cv::Mat& imgIn, cv::Mat& DFT_image)
{
  while (true)
  {
    cv::Mat H;
    int choice;
    float D0;
    menuPrompts();
    std::cin >> choice;
    if (choice == 0)
    {
      break;
    }
    std::cout << "Enter the desired D0 (0-100 makes sense):\n";
    std::cin >> D0;

    switch (choice)
    {
      case 1:
        H = construct_H(imgIn, "Ideal LP", static_cast<float>(D0));
        break;
      case 2:
        H = construct_H(imgIn, "Gaussian LP", static_cast<float>(D0));
        break;
      case 3:
        H = construct_H(imgIn, "Ideal HP", static_cast<float>(D0));
        break;
      case 4:
        H = construct_H(imgIn, "Gaussian HP", static_cast<float>(D0));
        break;
      case 5:
        H = construct_H(imgIn, "BandPass", static_cast<float>(D0));
        break;
      case 6:
        H = construct_H(imgIn, "Notch", static_cast<float>(D0));
        break;
      default:
        std::cerr << "Invalid choice\n";
        continue;
    }

    // Apply filtering and display the frequency domain
    cv::Mat filtered_img;
    filtering(DFT_image, filtered_img, H);
    show_dft_effect(filtered_img);

    // Doing a reversed DFT to visualize final effect
    cv::Mat imgOut = reverseDTF(filtered_img);
    imshow("Filtered Image", imgOut);

    cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);
    showHistogram(imgOut, "Filtered image histogram");

    if (cv::waitKey(0) == '0')
    {
      break;
    }
  }
}

int main()
{
  Mat imgIn = imread("../images/lena.png", IMREAD_GRAYSCALE);
  imshow("img", imgIn);
  waitKey();
  showHistogram(imgIn);
  waitKey();
  // Converting from 8-bit to float type suitable for DFT
  imgIn.convertTo(imgIn, CV_32F);

  // Calculate DFT
  Mat DFT_image;
  calculateDFT(imgIn, DFT_image);
  show_dft_effect(DFT_image);

  menuLoop(imgIn, DFT_image);

  return 0;
}

// WAVELETS IMPLEMENTATION
// int main(int ac, char** av)
// {
//     VideoCapture capture(0);
//     if (!capture.isOpened())
//     {
//         return 1;
//     }
//     return process(capture);
// }
