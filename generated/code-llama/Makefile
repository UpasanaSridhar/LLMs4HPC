
dgemm_avx2.x : generated.h ../../drivers/dgemm.cpp
	g++ -O3 -march=native -o dgemm_avx2.x ../../drivers/dgemm.cpp -I/home/upasanas/software/llms-hpc/KernelEval_Benchmark/generated/code-llama  -DAVX2=1

dgemm_avx512.x : generated.h ../../drivers/dgemm.cpp
	g++ -O3 -march=native -o dgemm_avx512.x ../../drivers/dgemm.cpp -I/home/upasanas/software/llms-hpc/KernelEval_Benchmark/generated/code-llama  -DAVX512=1


sgemm_avx512.x : generated.h ../../drivers/sgemm.cpp
	g++ -O3 -march=native -o sgemm_avx512.x ../../drivers/sgemm.cpp -I/home/upasanas/software/llms-hpc/KernelEval_Benchmark/generated/code-llama -DAVX512=1


sgemm_avx2.x : generated.h ../../drivers/sgemm.cpp
	g++ -O3 -march=native -o sgemm_avx2.x ../../drivers/sgemm.cpp -I/home/upasanas/software/llms-hpc/KernelEval_Benchmark/generated/code-llama -DAVX2=1



clean: 
	rm -f *.x