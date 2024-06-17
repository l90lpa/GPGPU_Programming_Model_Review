all:
	mkdir -p build

	cd ./build && cmake \
	-DKokkos_ENABLE_THREADS=ON \
	-DKokkos_ENABLE_CUDA=ON \
	..

	cd build && cmake --build . -- -j8

	cp ./build/matvec_kokkos .

clean:
	rm -r build matvec_kokkos