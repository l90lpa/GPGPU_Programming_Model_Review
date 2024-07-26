#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>


// Function to perform matrix-vector multiplication
template<typename ExecSpace, typename ViewType>
void MV(ViewType A,
        ViewType x,
        ViewType y,
        int rows,
        int cols) {


    using TeamHandle = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    using ScalarType = typename ViewType::value_type;

    const auto league_size = rows;
    const auto team_size = Kokkos::AUTO();

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(league_size, team_size),
        KOKKOS_LAMBDA (TeamHandle team) {
          ScalarType sum = 0;
          const long i = team.league_rank();
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, cols),
              [=] (const long j, ScalarType& sum) { sum += A[i * cols + j] * x[j]; },
              sum);
          y[i] = sum;
        });
}


int main(int argc, char* argv[]) {

  using FPType = float;

  Kokkos::initialize(argc, argv);
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

  { // <-- Scope being used to ensure that all Kokkos containers are
    //     deallocated before Kokkos::finalize is called.

    const int rows = 1024;
    const int cols = rows;

    Kokkos::View<FPType*> A_device("A", rows * cols); 
    Kokkos::View<FPType*> x_device("x", cols);
    Kokkos::View<FPType*> y_device("y", cols);

    // Initialize matrix
    Kokkos::parallel_for("InitMatrix", rows * cols,
      KOKKOS_LAMBDA (const long i) {
        A_device(i) = 2.0;
      }
    );

    // Initialize vector
    Kokkos::parallel_for("InitVector", cols,
      KOKKOS_LAMBDA (const long i) {
        x_device(i) = 1.0;
        y_device(i) = 0.0;
      }
    );

    MV<Kokkos::DefaultExecutionSpace>(A_device, x_device, y_device, rows, cols);

    MV<Kokkos::DefaultExecutionSpace>(A_device, y_device, y_device, rows, cols);

    auto y_host = Kokkos::create_mirror_view(y_device);    
    Kokkos::deep_copy(y_host, y_device);

    std::cout << "y(0) = " << y_host(0) << " " << ((y_host(0) == 4194304) ? "(pass)" : "(fail)") << std::endl;
  }

  Kokkos::finalize();

  return 0;
}
