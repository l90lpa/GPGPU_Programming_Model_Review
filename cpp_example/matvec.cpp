// #include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// Function to perform matrix-vector multiplication
template<typename ScalarType>
void MV(const std::vector<ScalarType>& A,
        const std::vector<ScalarType>& x,
        std::vector<ScalarType>& y,
        int rows,
        int cols) {

    for (int i = 0; i < rows; ++i) {
        ScalarType sum = 0;
        for (int j = 0; j < cols; ++j) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

int main() {
    
    const int rows = 1024;
    const int cols = rows;

    using FPType = float;

    const std::vector<FPType> A(rows * cols, 2.0);
    const std::vector<FPType> x(cols, 1.0);
    std::vector<FPType> y(cols, 0.0);

    MV(A, x, y, rows, cols);

    MV(A, y, y, rows, cols);

    std::cout << "y(0) = " << y[0] << " " << ((y[0] == 4194304) ? "(pass)" : "(fail)") << std::endl;

    return 0;
}