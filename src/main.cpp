#include <iostream>
#include <opencv2/highgui.hpp>

#include "helpers.hpp"
#include "image_processing.hpp"
#include "wavelets.hpp"

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
    int n;
    float epsilon;
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
      case 7: // Butterworth LP
        std::cout << "Enter the order n (Typical values are 1-5):\n";
        std::cin >> n;
        H = construct_H(imgIn, "Butterworth LP", static_cast<float>(D0), n);
        break;
      case 8: // Chebyshev LP
        std::cout << "Enter the order n (Typical values are 1-5):\n";
        std::cin >> n;
        std::cout << "Enter the ripple factor epsilon (Typical values are 0.1-0.5):\n";
        std::cin >> epsilon;
        H = construct_H(imgIn, "Chebyshev LP", static_cast<float>(D0), n, epsilon);
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
  cv::Mat imgIn;
  cv::Mat DFT_image;

  const int wdtIter = 2;
  const int brightnessScale = 1.5;
  imgIn = cv::imread("../images/lena.png", cv::IMREAD_GRAYSCALE);

  // Wavelets
  processWavelet(imgIn, wdtIter, brightnessScale);

  // DFT
  imshow("img", imgIn);
  cv::waitKey();
  showHistogram(imgIn);
  cv::waitKey();

  calculateDFT(imgIn, DFT_image);
  show_dft_effect(DFT_image);

  menuLoop(imgIn, DFT_image);

  return 0;
}
