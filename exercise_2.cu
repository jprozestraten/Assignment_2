#include <stdio.h>
#define ARRAY_SIZE 10000
#define TPB 256

__device__ float saxpy(float x, float y, float a)
{
  return a*x+y;
}

__global__ void saxpyKernel(float* x, float* y, float a)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = saxpy(x[i], y[i], a);
}


__host__ void saxpyCPU(float* x, float* y, float a)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		y[i] = a * x[i] + y[i];
	}
}

int main()
{
  
	// Declare a pointer for an array of floats
	float x_cpu[ARRAY_SIZE];
	float y_cpu[ARRAY_SIZE];
	float* x_gpu = 0;
	float* y_gpu = 0;
	float y_res[ARRAY_SIZE];
	const float a = 2;

	bool flag;

	// Array initialization
	for (int i = 0; i < ARRAY_SIZE; i++) {
		y_cpu[i] = i;
		x_cpu[i] = 1;
	}
  	
  	/* GPU CALCULATION */

	// Allocate device memory 
	cudaMalloc(&x_gpu, ARRAY_SIZE*sizeof(float));
	cudaMalloc(&y_gpu, ARRAY_SIZE*sizeof(float));

	// Copy the arrays from CPU to GPU
	cudaMemcpy(x_gpu, x_cpu, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y_cpu, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);


	// Launch kernel to compute and store distance values
	saxpyKernel<<<(ARRAY_SIZE+TPB-1) / TPB, TPB>>>(x_gpu, y_gpu, a);

  	cudaMemcpy(y_res, y_gpu, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  	printf("Computing SAXPY on the GPU… Done!\n");


  	/* CPU CALCULATION */
	saxpyCPU(x_cpu, y_cpu, a);

	printf("Computing SAXPY on the CPU… Done!\n");

	/* COMPARE THE RESULTS */
	flag = 1;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if(y_res[i] != y_cpu[i]) {
			flag = 0;
			break;
		}
	}
	printf("Comparing the output for each implementation… ");
	if (flag)
	{
		printf("Correct!\n");
	} else {
		printf("Incorrect\n");
	}

	cudaFree(x_gpu); // Free the memory
	cudaFree(y_gpu);
	return 0;
}



