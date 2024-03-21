#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

// MACROS
#define PI 3.1415926536

// TYPEDEF's
typedef complex<double> Complex; // Complex number representation

// Function to reverse bits (same as provided previously)
unsigned int bitReverse(unsigned int x, int log2n)
{
  int n = 0;
  int mask = 0x1;
  for (int i = 0; i < log2n; i++)
  {
    n <<= 1;
    n |= (x & 1);
    x >>= 1;
  }
  return n;
}

// Function to perform 1D FFT (adapted from provided code)
void fft(vector<Complex>& data, bool inverse)
{
  const int n = data.size();
  int log2n = 0;
  while (n >> log2n != 1)
  {
    log2n++;
  }

  vector<Complex> temp(n);

  Complex J(0, (inverse ? -1 : 1));

  for (int i = 0; i < n; ++i)
  {
    temp[bitReverse(i, log2n)] = data[i];
  }

  for (int s = 1; s <= log2n; ++s)
  {
    int m = 1 << s;
    int m2 = m >> 1;
    Complex w(1, 0);
    Complex wm = exp(J * (PI / m2));
    for (int j = 0; j < m2; ++j)
    {
      for (int k = j; k < n; k += m)
      {
        Complex t = w * temp[k + m2];
        Complex u = temp[k];
        temp[k] = u + t;
        temp[k + m2] = u - t;
      }
      w *= wm;
    }
  }

  if (inverse)
  {
    for (int i = 0; i < n; ++i)
    {
      temp[i] /= n;
    }
  }
  copy(temp.begin(), temp.end(), data.begin());
}

// Function to perform 2D FFT
void fft2d(vector<vector<Complex>>& data, bool inverse)
{
  const int rows = data.size();
  const int cols = data[0].size();

  // Apply 1D FFT to each row
  for (int i = 0; i < rows; ++i)
  {
    fft(data[i], inverse);
  }

  // Transpose the data
  vector<vector<Complex>> transposed(cols, vector<Complex>(rows));
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      transposed[j][i] = data[i][j];
    }
  }

  // Apply 1D FFT to each column of the transposed data
  for (int i = 0; i < cols; ++i)
  {
    fft(transposed[i], inverse);
  }

  // Transpose back if not in inverse mode
  if (!inverse)
  {
    data = transposed;
  }
}

// Function to fill the 2D matrix with sample data
void fillData(vector<vector<Complex>>& data, int rows, int cols)
{
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      data[i][j] = Complex(i * j, 0.1 * i + 0.2 * j); // Example initialization
    }
  }
}

// Function to print the 2D complex matrix
void printMatrix(const vector<vector<Complex>>& data)
{
  // Check if there are no rows
  if (data.empty())
  {
    cout << "Empty matrix" << endl;
    return;
  }
  // Determine the maximum width for real and imaginary parts
  int realWidth = 0, imagWidth = 0;
  for (const auto& row : data)
  {
    for (const auto& element : row)
    {
      realWidth = std::max(realWidth, static_cast<int>(std::to_string(element.real()).size()));
      imagWidth = std::max(imagWidth, static_cast<int>(std::to_string(element.imag()).size()));
    }
  }
  int width = std::max(realWidth, imagWidth) + 2;

  // Print header row
  for (int i = 0; i < data[0].size(); ++i)
  {
    cout << std::setw(width) << "Real"
         << " | " << std::setw(width) << "Imag"
         << " | ";
  }
  cout << endl;

  // Print data rows
  for (const auto& row : data)
  {
    for (const auto& element : row)
    {
      cout << std::setw(width) << element.real() << " | " << std::setw(width) << element.imag() << " | ";
    }
    cout << endl;
  }
}
