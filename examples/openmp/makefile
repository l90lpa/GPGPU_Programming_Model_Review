matvec_omp: matvec_omp.cpp
	nvc++ -mp=gpu -Minfo=mp -std=c++17 -o matvec_omp matvec_omp.cpp

clean:
	rm matvec_omp