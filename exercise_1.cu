#include <stdio.h>

__global__ void helloKernel()
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("Hello World! My threadId is %d \n", i);
}

int main()
{ 
  // Launch kernel to print 
  helloKernel<<<1, 256>>>();
  cudaDeviceSynchronize();
  return 0;
}
