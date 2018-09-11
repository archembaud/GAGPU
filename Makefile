all: GPU CPU
	nvcc cpu_main.o gpu_main.o -o GAGPU.run

GPU:
	nvcc gpu_main.cu -c

CPU:
	g++ cpu_main.cpp -c

