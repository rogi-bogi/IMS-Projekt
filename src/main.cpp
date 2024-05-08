#include <iostream>
#include <cmath>

#include "FFT.hpp"
#include "image_processing.hpp"
#include "helpers.hpp"

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

int main()
{
	Mat imgIn = imread("../images/lena.png", IMREAD_GRAYSCALE);
  	imshow("img", imgIn);

 	// Converting from 8-bit to float type suitable for DFT
	imgIn.convertTo(imgIn, CV_32F);

	// Calculate DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);
 	show_dft_effect(DFT_image);

	// Construct H's (filter matrices)
	Mat H_1, H_2, H_3, H_4, H_5, H_6;
	H_1 = construct_H(imgIn, "Ideal LP", 80); // Low-pass filter
	H_2 = construct_H(imgIn, "Gaussian LP", 80); // Gaussian low-pass filter
	H_3 = construct_H(imgIn, "Ideal HP", 50); // High-pass filter
	H_4 = construct_H(imgIn, "Gaussian HP", 80); // Gaussian high-pass filter
	H_5 = construct_H(imgIn, "BandPass", 50); // Band-pass filter
	H_6 = construct_H(imgIn, "Notch", 50); // Notch filter

	// Apply filtering and display the frequency domains
	Mat H1_filtered_img, H2_filtered_img, H3_filtered_img, H4_filtered_img, H5_filtered_img, H6_filtered_img;

	filtering(DFT_image, H1_filtered_img, H_1);
	show_dft_effect(H1_filtered_img);

	filtering(DFT_image, H2_filtered_img, H_2);
	show_dft_effect(H2_filtered_img);

	filtering(DFT_image, H3_filtered_img, H_3);
	show_dft_effect(H3_filtered_img);

	filtering(DFT_image, H4_filtered_img, H_4);
	show_dft_effect(H4_filtered_img);

	filtering(DFT_image, H5_filtered_img, H_5);
	show_dft_effect(H5_filtered_img);

	filtering(DFT_image, H6_filtered_img, H_6);
	show_dft_effect(H6_filtered_img);

	// Doing a reversed DFT to visualize final effect
	Mat imgOut_1 = reverseDTF(H1_filtered_img);
	imshow("Ideal LP filter", imgOut_1);

	Mat imgOut_2 = reverseDTF(H2_filtered_img);
	imshow("Gaussian LP filter", imgOut_2);

	Mat imgOut_3 = reverseDTF(H3_filtered_img);
	imshow("Ideal HP filter", imgOut_3);

	Mat imgOut_4 = reverseDTF(H4_filtered_img);
	imshow("Gaussian HP filter", imgOut_4);

	Mat imgOut_5 = reverseDTF(H5_filtered_img);
	imshow("Band Pass filter", imgOut_5);

	Mat imgOut_6 = reverseDTF(H6_filtered_img);
	imshow("Notch filter", imgOut_6);

	waitKey(0);
	return 0;
}
