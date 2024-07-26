all:
	mkdir -p build

	cd build && cmake \
	-DCMAKE_BUILD_TYPE=Debug \
	-DBLT_CXX_STD=c++17 \
	-DENABLE_CUDA=ON \
	-DENABLE_OPENMP=ON \
	-DENABLE_TESTS=OFF \
	-DENABLE_EXAMPLES=OFF \
	-DRAJA_ENABLE_EXERCISES=OFF \
	-DCMAKE_C_COMPILER=gcc \
	-DCMAKE_CXX_COMPILER=g++ \
	-DCMAKE_CUDA_COMPILER=nvcc \
	-DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3 \
	-DCUDA_ARCH=native \
	..

	cd build && cmake --build . -- -j8
	cp ./build/bin/matvec_raja .

clean:
	rm -r build matvec_raja