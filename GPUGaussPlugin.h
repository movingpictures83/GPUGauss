#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUGaussPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
		float* a;
		int N;
                std::map<std::string, std::string> parameters;
};

__global__ void gpu_zeroColumn(float* a, int N, int i) {
    int j, k;
    j = blockIdx.x;  //row
    k = threadIdx.x; //col
    if (j > i  && j < N && k >= i && k < N){
        float denominator = a[i * N + i];
        float val = - (a[j * N + i] / denominator);
        __syncthreads();
        a[j * N + k] = val * a[i * N + k] + a[j * N + k];
    }
} 
