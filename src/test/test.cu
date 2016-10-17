#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

__device__ int va;

__global__ void vecMul(float *a, float *b, float* re, int n)
{
    int	id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>n) return;
	re[id] = a[id]*b[id];
}


__global__ void One(float *a, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>n) return;
	a[id]=10.0;
}

__global__ void print1(float *a, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	printf("aaa");
	if(id>n) return;
	printf("\na[%d] = %f\n",id,a[id]);
}

__global__ void printva()
{
//	printf("\n gpu side:");
	va++;
}

class aaa
{
	public:
		aaa(){};
		~aaa(){};
			
	int a;
	int *b;
};

double get_time()
{
  struct timeval t;
  struct timezone tzp;
  gettimeofday(&t, &tzp);
  return t.tv_sec + t.tv_usec*1e-6;
}


int main()
{
/*	int a = 1;
	int *b;
	aaa* faaa = new aaa();
	cudaMalloc((void**)&(faaa->b),sizeof(int));
	printf("%lf",get_time());
	double start = get_time();
//	double start = get_time();
	cudaMemcpy(faaa->b,&(faaa->a),sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(&(faaa->a),faaa->b,sizeof(int),cudaMemcpyDeviceToHost);
//	cudaMemcpyToSymbol(b,&a,sizeof(int));
	printf("\nelapsed time:%lf\n",get_time()-start);
	start = get_time();
	cudaMemcpyToSymbol(va,&a,sizeof(int));
	cudaMemcpyFromSymbol(&a,va,sizeof(int));
	printf("back transfer time:%lf\n",get_time() - start);
	printf("done!");
	printva<<<1,100>>>();
	printva<<<1,100>>>();
	cudaMemcpyFromSymbol(&a,va,sizeof(int));
	
	printf("a value: %d",a);
	getchar();*/
	float *a;
	float *b;
	float *re;
	float h_a[10];
	cudaMalloc((void**)&(a), 10*sizeof(float));
	cudaMalloc((void**)&(b), 10*sizeof(float));
	cudaMalloc((void**)&(re), 10*sizeof(float));

	One<<<1,10>>>(a,10);
	One<<<1,10>>>(b,10);
	vecMul<<<1,10>>>(a,b,a,10);
	printf("toprint value\n");
	print1<<<1,10>>>(a,10);

	cudaMemcpy(h_a,a,10*sizeof(float),cudaMemcpyDeviceToHost);

	printf("h_a = %f",h_a[0]);
	getchar();
}

