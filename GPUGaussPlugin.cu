#include <emmintrin.h>
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "GPUGaussPlugin.h"

void GPUGaussPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 N = atoi(parameters["N"].c_str());
 a = (float*) malloc(N*N*sizeof(float));
 int M = N * N;
 std::ifstream myinput((std::string(PluginManager::prefix())+parameters["matrix"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M; ++i) {
	int k;
	myinput >> k;
        a[i] = k;
 }
}

void GPUGaussPlugin::run() {
    int i;
    for (i = 32; i <= 1024; i += 32){
        if (i >= N) { break; }
    }
    int numThreads = i;
    int numCores = numThreads;

    float* gpuA;

    // Allocate enough memory on the GPU
    cudaMalloc(&gpuA, N * N * sizeof (float)); 

    // Copy array from CPU to GPU
    cudaMemcpy(gpuA, a, N * N * sizeof (float), cudaMemcpyHostToDevice); 


    for (i = 0; i < N; i++){
        gpu_zeroColumn <<<numCores, numThreads>>>(gpuA, N, i); 
    }

    // Copy array from GPU to CPU
    cudaMemcpy(a, gpuA, N * N * sizeof (float), cudaMemcpyDeviceToHost); 

    // Free the memory on the GPU
    cudaFree(&gpuA); 
}

void GPUGaussPlugin::output(std::string file) {
	std::ofstream outfile(file.c_str(), std::ios::out);
        int i, j;
        for (i = 0; i < N; ++i){
            for (j = 0; j < N; ++j){
		outfile << (int) a[i*N+j];//std::setprecision(0) << a[i*N+j];
		if (j != N-1)
			outfile << "\t";
		else
			outfile << "\n";
            }
	}
	free(a);
}



PluginProxy<GPUGaussPlugin> GPUGaussPluginProxy = PluginProxy<GPUGaussPlugin>("GPUGauss", PluginManager::getInstance());


