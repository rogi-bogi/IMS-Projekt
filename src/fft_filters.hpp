#pragma once

#include <opencv2/opencv.hpp>

void idealLpFilter(cv::Mat& scr, cv::Mat& H, float D, float D0)
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

void gaussianLpFilter(cv::Mat& scr, cv::Mat& H, float D, float D0)
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

void idealHpFilter(cv::Mat& scr, cv::Mat& H, float D, float D0)
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

void gaussianHpFilter(cv::Mat& scr, cv::Mat& H, float D, float D0)
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

void bandPassFilter(cv::Mat& scr, cv::Mat& H, float D, float D0)
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

void notchFilter(cv::Mat& scr, cv::Mat& H, float D, float D0)
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

void butterworthLpFilter(cv::Mat& scr, cv::Mat& H, float D, float D0, int n)
{
    for (int u = 0; u < H.rows; u++)
    {
        for (int v = 0; v < H.cols; v++)
        {
            D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
            H.at<float>(u, v) = 1 / (1 + pow(D / D0, 2 * n));
        }
    }
}

void chebyshevLpFilter(cv::Mat& scr, cv::Mat& H, float D, float D0, float epsilon, int n)
{
    for (int u = 0; u < H.rows; u++)
    {
        for (int v = 0; v < H.cols; v++)
        {
            D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
            float term = pow((D / D0), n);
            float chebyshevTerm = 1 + pow(epsilon * cosh(term), 2);
            H.at<float>(u, v) = 1 / sqrt(chebyshevTerm);
        }
    }
}