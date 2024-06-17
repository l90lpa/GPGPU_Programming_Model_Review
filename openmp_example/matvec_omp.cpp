#include <omp.h>

#include <iostream>
#include <cassert>
#include <vector>

// Function to perform matrix-vector multiplication
template<typename ScalarType>
void MV(const ScalarType * const A,
        const ScalarType * const x,
        ScalarType * const y,
        int rows,
        int cols) {

    #pragma omp target teams loop map(to: A[0:rows*cols], x[0:cols]) map(from: y[0:cols])
    for (int i = 0; i < rows; ++i) {
        ScalarType sum = 0;
        #pragma omp loop reduction(+:sum)
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

    const FPType* const A_ = A.data();
    const FPType* const x_ = x.data();
    FPType* const y_ = y.data();

    // #pragma omp target enter data map(to: _A[0:rows*cols], _x[0:cols]) map(alloc: _y[0:cols])

    // Perform matrix-vector multiplication
    MV(A_, x_, y_, rows, cols);

    // Perform matrix-vector multiplication
    MV(A_, y_, y_, rows, cols);

    // #pragma omp target exit data map(from: _y[0:cols])

    std::cout << "y(0) = " << y[0] << " " << ((y[0] == 4194304) ? "(pass)" : "(fail)") << std::endl;

    return 0;
}