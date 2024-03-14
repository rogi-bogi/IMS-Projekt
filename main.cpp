//
// Created by Kacper on 12.03.2024.
//

// INCLUDES
#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;

// MACROS
#define PI              3.1415926536

// Function to reverse bits (same as provided previously)
unsigned int bitReverse(unsigned int x, int log2n) {
    int n = 0;
    int mask = 0x1;
    for (int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Complex number representation
typedef complex<double> Complex;

// Function to perform 1D FFT (adapted from provided code)
void fft(vector<Complex>& data, bool inverse) {
    const int n = data.size();
    int log2n = 0;
    while (n >> log2n != 1) {
        log2n++;
    }

    vector<Complex> temp(n);

    Complex J(0, (inverse ? -1 : 1));

    for (int i = 0; i < n; ++i) {
        temp[bitReverse(i, log2n)] = data[i];
    }

    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        Complex w(1, 0);
        Complex wm = exp(J * (PI / m2));
        for (int j = 0; j < m2; ++j) {
            for (int k = j; k < n; k += m) {
                Complex t = w * temp[k + m2];
                Complex u = temp[k];
                temp[k] = u + t;
                temp[k + m2] = u - t;
            }
            w *= wm;
        }
    }

    if (inverse) {
        for (int i = 0; i < n; ++i) {
            temp[i] /= n;
        }
    }
    copy(temp.begin(), temp.end(), data.begin());
}

// Function to perform 2D FFT
void fft2d(vector<vector<Complex>>& data, bool inverse) {
    const int rows = data.size();
    const int cols = data[0].size();

    // Apply 1D FFT to each row
    for (int i = 0; i < rows; ++i) {
        fft(data[i], inverse);
    }

    // Transpose the data
    vector<vector<Complex>> transposed(cols, vector<Complex>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = data[i][j];
        }
    }

    // Apply 1D FFT to each column of the transposed data
    for (int i = 0; i < cols; ++i) {
        fft(transposed[i], inverse);
    }

    // Transpose back if not in inverse mode
    if (!inverse) {
        data = transposed;
    }
}

// Function to fill the 2D matrix with sample data
void fillData(vector<vector<Complex>>& data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = Complex(i * j, 0.1 * i + 0.2 * j); // Example initialization
        }
    }
}

// Function to print the 2D complex matrix
#include <iomanip>

void printMatrix(const vector<vector<Complex>>& data) {
//    // Determine the maximum width for real and imaginary parts
//    int realWidth = 0, imagWidth = 0;
//    for (const auto& row : data) {
//        for (const auto& element : row) {
//            realWidth = std::max(realWidth, static_cast<int>(std::to_string(element.real()).size()));
//            imagWidth = std::max(imagWidth, static_cast<int>(std::to_string(element.imag()).size()));
//        }
//    }

    // Print header row
    cout << "  | ";
    for (int i = 0; i < data[0].size(); ++i) {
        cout << std::setw(10) << "Real" << " | " << std::setw(10) << "Imag" << " | ";
    }
    cout << endl;

    // Print data rows
    for (const auto& row : data) {
        for (const auto& element : row) {
            cout << std::setw(10) << element.real() << " | " << std::setw(10) << element.imag() << " | ";
        }
        cout << endl;
    }
}

int main() {
    // Example usage
    int rows = 4;
    int cols = 4;

    // Create a 2D vector to store complex numbers
    vector<vector<Complex>> data(rows, vector<Complex>(cols));

    // Fill the data with sample values (replace with your actual data if needed)
    fillData(data, rows, cols);

    // Print the original data matrix
    cout << "Original data:" << endl;
    printMatrix(data);

    // Perform 2D FFT (forward transform)
    fft2d(data, false);

    // Print the data matrix after the 2D FFT (frequency domain)
    cout << "Data after 2D FFT (frequency domain):" << endl;
    printMatrix(data);

    // Optional: Perform inverse FFT (if needed)
    // fft2d(data, true); // Inverse FFT

    return 0;
}
