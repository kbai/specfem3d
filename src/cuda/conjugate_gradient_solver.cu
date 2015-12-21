#include <cuda.h> 
#include <cuda_runtime.h>
#include <stdio.h>
#include "conjugate_gradient_solver.hh"

__global__ void sumall(realw* array, int size, realw* sum)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
	if( id >= size) return;
//	printf("special value:%f",array[id]);
	atomicAdd(sum, array[id]);
	return;
}

__global__ void testarray(realw* array, int size)
{

	int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
	if(id >= size) return;
//	array[id] = 1.000;
//	if(id == 1) printf("\narrayvalue:%d:%f\n",id,array[id]);

}	

__global__ void compute_forces(realw* displ, realw* forces, bool* maskx, bool* maskax)
{



}	

conjugate_gradient::conjugate_gradient(int NELE, realw* pdis, realw* pload, bool* Xsetfalse, bool* AXsetfalse, realw* gpu_displ, realw* gpu_force,  bool precon ,int proc_num)
{
	this->NSPEC = NELE;
	/** reference to the outside array*/
	this->h_displ = pdis;
	this->h_load = pload;
	this->myrank = proc_num;

	this->h_MASKX = Xsetfalse;
	this->h_MASKAX = AXsetfalse;
    
	this->h_Mprecon = new realw[3*NSPEC];
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_pdire)), NSPEC*3*sizeof(realw)), 50001);
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_residue)), NSPEC*3*sizeof(realw)), 50004);
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_precon)), NSPEC*3*sizeof(realw)), 50010);
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_sum)),sizeof(realw)),50002);
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_norm)),sizeof(realw)),50003);
	print_CUDA_error_if_any(cudaMemset(this->d_sum, 0 , sizeof(realw)) ,50006);
	print_CUDA_error_if_any(cudaMemset(this->d_norm, 0 , sizeof(realw)) ,50007);
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_MASKAX)), NSPEC*3*sizeof(bool)), 50011);
	print_CUDA_error_if_any(cudaMalloc((void**)(&(this->d_MASKX)), NSPEC*3*sizeof(bool)), 50012);
	print_CUDA_error_if_any(cudaMemcpy(this->d_MASKX, this->h_MASKX, NSPEC*3*sizeof(bool), cudaMemcpyHostToDevice),50015);
	print_CUDA_error_if_any(cudaMemcpy(this->d_MASKAX, this->h_MASKAX, NSPEC*3*sizeof(bool), cudaMemcpyHostToDevice),500016);
	this->d_displ = gpu_displ;
	this->d_force = gpu_force;
	if(precon)
		this->compute_precon();
	print_CUDA_error_if_any(cudaMemcpy(this->d_precon, this->h_Mprecon, NSPEC*3*sizeof(realw), cudaMemcpyHostToDevice),50005);

}

conjugate_gradient::~conjugate_gradient()
{}

void conjugate_gradient::compute_precon()
{
	compute_diagonal_(this->h_Mprecon);   
}

void conjugate_gradient::checkfield()
{
//	std::cout<<std::endl<<"check:"<<h_displ[0]<<" "<<h_displ[1]<<" "<<h_displ[2]<<std::endl;
//	std::cout<<std::endl<<"check precon1:"<<h_Mprecon[0]<<" "<<h_Mprecon[1]<<" "<<h_Mprecon[2]<<std::endl;
//	std::cout<<std::endl<<"check precon2:"<<h_load[0]<<" "<<h_load[1]<<" "<<h_load[2]<<std::endl;
//	std::cout<<"number of points:"<<NSPEC<<std::endl;
	this->gpu_init();
   

}

	
void conjugate_gradient::gpu_init()
{
	int num_blocks_x, num_blocks_y;
	get_blocks_xy((int)ceil((double)(3*NSPEC)/128.0),&num_blocks_x, &num_blocks_y);
	dim3 grid(num_blocks_x,num_blocks_y);
	std::cout<<num_blocks_x<<" "<<num_blocks_y << std::endl;
	dim3 threads(128,1,1);
    testarray<<<grid,threads>>>(this->d_displ, 3*NSPEC);

//	printf("finishes!\n");
}

void conjugate_gradient::sum()
{
	int num_blocks_x, num_blocks_y;
	get_blocks_xy((int)ceil((double)NSPEC/128.0),&num_blocks_x, &num_blocks_y);
	dim3 grid(num_blocks_x,num_blocks_y);
	std::cout<<num_blocks_x<<" "<<num_blocks_y << std::endl;
	dim3 threads(128,1,1);
    sumall<<<grid,threads>>>(this->d_displ, NSPEC, this->d_sum);
	print_CUDA_error_if_any(cudaMemcpy(&(this->h_sum),this->d_sum,sizeof(realw),cudaMemcpyDeviceToHost),50008);
//	printf("\n the sum of the array: %f::%d\n",(this->h_sum),this->NSPEC);

}

void conjugate_gradient::compute_forces()
{

	compute_fault_gpu_(this->d_force, this->d_displ, this->d_MASKX, this->d_MASKAX);

	print_CUDA_error_if_any(cudaMemcpy((this->h_load),this->d_force,NSPEC*3*sizeof(realw),cudaMemcpyDeviceToHost),50018);


//	printf("\nmyrank is : %d\n",this->myrank);

	if(this->myrank == 31) 
	{
      for(int i = 9000;i<=9030;i+=3) printf("accel at %d:%f",i,h_load[i]); 
	}
}
