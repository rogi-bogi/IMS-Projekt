#pragma once

#include "opencv2/opencv.hpp"

// Filter type
#define NONE 0   // no filter
#define HARD 1   // hard shrinkage
#define SOFT 2   // soft shrinkage
#define GARROT 3 // garrot filter

// signum function
float sgn(float x)
{
  if (x == 0)
    return 0;
  return (x > 0) ? 1 : -1;
}

// Soft shrinkage
float soft_shrink(float d, float T)
{
  if (fabs(d) > T)
  {
    return sgn(d) * (fabs(d) - T);
  }
  else
  {
    return 0;
  }
}

// Hard shrinkage
float hard_shrink(float d, float T) { return (fabs(d) > T) ? d : 0; }

// Garrot shrinkage
float Garrot_shrink(float d, float T) { return (fabs(d) > T) ? d - ((T * T) / d) : 0; }

// Wavelet transform
static void cvHaarWavelet(cv::Mat& src, cv::Mat& dst, int NIter)
{
  float c, dh, dv, dd;
  assert(src.type() == CV_32FC1);
  assert(dst.type() == CV_32FC1);
  int width = src.cols;
  int height = src.rows;
  for (int k = 0; k < NIter; k++)
  {
    for (int y = 0; y < (height >> (k + 1)); y++)
    {
      for (int x = 0; x < (width >> (k + 1)); x++)
      {
        c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) +
             src.at<float>(2 * y + 1, 2 * x + 1)) *
            0.5;
        dst.at<float>(y, x) = c;

        dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) -
              src.at<float>(2 * y + 1, 2 * x + 1)) *
             0.5;
        dst.at<float>(y, x + (width >> (k + 1))) = dh;

        dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) -
              src.at<float>(2 * y + 1, 2 * x + 1)) *
             0.5;
        dst.at<float>(y + (height >> (k + 1)), x) = dv;

        dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) +
              src.at<float>(2 * y + 1, 2 * x + 1)) *
             0.5;
        dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
      }
    }
    dst.copyTo(src);
  }
}

// Inverse wavelet transform
static void cvInvHaarWavelet(cv::Mat& src, cv::Mat& dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50)
{
  float c, dh, dv, dd;
  assert(src.type() == CV_32FC1);
  assert(dst.type() == CV_32FC1);
  int width = src.cols;
  int height = src.rows;
  for (int k = NIter; k > 0; k--)
  {
    for (int y = 0; y < (height >> k); y++)
    {
      for (int x = 0; x < (width >> k); x++)
      {
        c = src.at<float>(y, x);
        dh = src.at<float>(y, x + (width >> k));
        dv = src.at<float>(y + (height >> k), x);
        dd = src.at<float>(y + (height >> k), x + (width >> k));

        // Shrinkage
        switch (SHRINKAGE_TYPE)
        {
          case HARD:
            dh = hard_shrink(dh, SHRINKAGE_T);
            dv = hard_shrink(dv, SHRINKAGE_T);
            dd = hard_shrink(dd, SHRINKAGE_T);
            break;
          case SOFT:
            dh = soft_shrink(dh, SHRINKAGE_T);
            dv = soft_shrink(dv, SHRINKAGE_T);
            dd = soft_shrink(dd, SHRINKAGE_T);
            break;
          case GARROT:
            dh = Garrot_shrink(dh, SHRINKAGE_T);
            dv = Garrot_shrink(dv, SHRINKAGE_T);
            dd = Garrot_shrink(dd, SHRINKAGE_T);
            break;
        }

        dst.at<float>(y * 2, x * 2) = 0.5 * (c + dh + dv + dd);
        dst.at<float>(y * 2, x * 2 + 1) = 0.5 * (c - dh + dv - dd);
        dst.at<float>(y * 2 + 1, x * 2) = 0.5 * (c + dh - dv - dd);
        dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5 * (c - dh - dv + dd);
      }
    }
    cv::Mat C = src(cv::Rect(0, 0, width >> (k - 1), height >> (k - 1)));
    cv::Mat D = dst(cv::Rect(0, 0, width >> (k - 1), height >> (k - 1)));
    D.copyTo(C);
  }
}

void processWavelet(const cv::Mat& img, const int numIter = 3, const int scaleFactor = 1)
{
  cv::Mat Src, Dst, Temp, Filtered;
  // Converting from 8-bit to float type suitable for DFT and Wavelets operations
  img.convertTo(Src, CV_32F);

  Dst = cv::Mat(Src.size(), CV_32FC1);
  Temp = cv::Mat(Src.size(), CV_32FC1);
  Filtered = cv::Mat(Src.size(), CV_32FC1);

  cvHaarWavelet(Src, Dst, numIter);

  Dst.copyTo(Temp);

  cvInvHaarWavelet(Temp, Filtered, numIter, GARROT, 30);

  double M = 0, m = 0;

  minMaxLoc(Dst, &m, &M);
  if ((M - m) > 0)
  {
    Dst = (Dst - m) * (255.0 / (M - m));
  }

  minMaxLoc(Filtered, &m, &M);
  if ((M - m) > 0)
  {
    Filtered = (Filtered - m) * (255.0 / (M - m));
  }

  // Convert back to 8-bit for display
  Dst.convertTo(Dst, CV_8UC1);
  Filtered.convertTo(Filtered, CV_8UC1);

  // Enhance visibility by scaling to 0-255 range
  Dst = Dst * scaleFactor;
  Filtered = Filtered * scaleFactor;

  // Display images
  imshow("Original", img);
  imshow("Wavelet Coefficients", Dst);
  imshow("Filtered (Garrot)", Filtered);

  // Wait for a key press indefinitely
  cv::waitKey();
}
