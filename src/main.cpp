#include <iostream>
#include <cmath>

#include "FFT.hpp"
#include "image_processing.hpp"
#include "helpers.hpp"

using namespace cv;

//TODO: cv::butterworthFilter or cv::chebyshevFilter ??
//TODO: Custom filters based on what we have (just like built-ins)
//TODO: Export the functions to hpp

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

void calculateDFT(Mat &scr, Mat &dst)
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

void fftshift(const Mat &input_img, Mat &output_img)
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

// Basically creates a filtering matrix for Frequency Domain
Mat construct_H(Mat &scr, String type, float D0)
{
	Mat H(scr.size(), CV_32F, Scalar(1));
	float D = 0;
	if (type == "Ideal")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int  v = 0; v < H.cols; v++)
			{
				D = sqrt((u - scr.rows / 2)*(u - scr.rows / 2) + (v - scr.cols / 2)*(v - scr.cols / 2));
				if (D > D0)
				{
					H.at<float>(u, v) = 0;
				}
			}
		}
	}
	else if (type == "Gaussian")
	{
		for (int  u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D = sqrt((u - scr.rows / 2)*(u - scr.rows / 2) + (v - scr.cols / 2)*(v - scr.cols / 2));
				H.at<float>(u, v) = exp(-D*D / (2 * D0*D0));
			}
		}
	}
  return H;
}

void filtering(Mat &scr, Mat &dst, Mat &H)
{
	fftshift(H, H);
	Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

	Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	split(scr, planes_dft);

	Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
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
	Mat planes[] = { real, imaginary };

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

int main()
{
	Mat imgIn = imread("../images/lena.png", IMREAD_GRAYSCALE);
  imshow("img", imgIn);

  // Converting to float type suitable for DFT
	imgIn.convertTo(imgIn, CV_32F);

	// calculate DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

  show_dft_effect(DFT_image);

	// Construct H (filter matrix)
	Mat H_1, H_2;
	H_1 = construct_H(imgIn, "Ideal", 80);
  H_2 = construct_H(imgIn, "Gaussian", 80);

	// Filtering - merging two frequency domain matrices 
	Mat complexIH_1, complexIH_2;
	filtering(DFT_image, complexIH_1, H_1);
  show_dft_effect(complexIH_1);

  filtering(DFT_image, complexIH_2, H_2);
  show_dft_effect(complexIH_2);

  // Doing a reversed DFT to visualize effect
  Mat imgOut_1 = reverseDTF(complexIH_1);
  imshow("Ideal filter", imgOut_1);

  Mat imgOut_2 = reverseDTF(complexIH_2);
  imshow("Gaussian filter", imgOut_2);

  waitKey(0);
	return 0;
}
