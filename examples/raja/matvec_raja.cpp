#include <RAJA/RAJA.hpp>
#include <chai/ManagedArray.hpp>

#include <iostream>

// Define RAJA execution policy
#if defined(RAJA_ENABLE_CUDA)
using PARA_EXEC_POLICY = RAJA::cuda_exec<256>;
#define PARA_EXEC_POLICY_STR "RAJA::cuda_exec<256>"
#elif defined(RAJA_ENABLE_HIP)
using PARA_EXEC_POLICY = RAJA::hip_exec<256>;
#define PARA_EXEC_POLICY_STR "RAJA::hip_exec<256>"
#else
using PARA_EXEC_POLICY = RAJA::seq_exec;
#define PARA_EXEC_POLICY_STR "RAJA::seq_exec"
#endif


// Function to perform matrix-vector multiplication
template<typename ScalarType>
void MV(const chai::ManagedArray<ScalarType>& A,
        const chai::ManagedArray<ScalarType>& x,
        chai::ManagedArray<ScalarType>& y,
        int rows,
        int cols) {

    RAJA::forall<PARA_EXEC_POLICY>(
        RAJA::RangeSegment(0, rows),
        [=] RAJA_DEVICE (int i) {
        ScalarType sum = 0;
        for (int j = 0; j < cols; ++j) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    });
}


int main() {
    const int rows = 1024;
    const int cols = rows;

    std::cout << "PARA_EXEC_POLICY = " << PARA_EXEC_POLICY_STR << std::endl;

    using FPType = float;
   
    chai::ManagedArray<FPType> A(rows * cols);
    chai::ManagedArray<FPType> x(cols);
    chai::ManagedArray<FPType> y(rows);

    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, rows * cols), [=] (int i) {
        A[i] = 2.0;
    });

    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, cols), [=] (int i) {
        x[i] = 1.0;
        y[i] = 0.0;
    });

    MV(A, x, y, rows, cols);

    MV(A, y, y, rows, cols);

    y.move(chai::CPU);
    
    std::cout << "y(0) = " << y[0] << " " << ((y[0] == 4194304) ? "(pass)" : "(fail)") << std::endl;

    return 0;
}