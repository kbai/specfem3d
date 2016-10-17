#include <stdio.h>
#include <config.h>
#include <iostream>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include "mesh_constants_cuda.h"
#include "conjugate_gradient_solver.hh"
extern "C"
void FC_FUNC_(passhandle,
		      PASSHANDLE)(long* Mesh_pointer,long* CG_pointer, int* NELE, realw* disp, realw* pload,realw* recorder, int*Xsetfalse, int* AXsetfalse, int* myrank, int* MPI_repeat, realw* restri_op  )
{
	printf("size used here: int : %d, realw: %d",sizeof(int),sizeof(realw));
	Mesh* mp = (Mesh*)(*Mesh_pointer);
	printf("\nMYRANK:%d\n",*myrank);
	//conjugate_gradient *newCG;
	conjugate_gradient *newCG = new conjugate_gradient(*NELE , disp, pload, Xsetfalse, AXsetfalse, mp->d_solution, mp->d_accel, true, *myrank, recorder, MPI_repeat,restri_op, mp->d_rmassx);
	cudaMemcpy(mp->d_displ, disp, 3*(*NELE)*sizeof(realw),cudaMemcpyHostToDevice);
	MPI_Barrier(MPI_COMM_WORLD);
	*CG_pointer = (long)(newCG); 

//	cuProfilerInitialize();
	double start = get_time();

	cudaProfilerStart();
	

	cudaProfilerStop();


	printf("\n time elapsed: %lf ",get_time()-start);
//	abort();
}	



extern "C"
void FC_FUNC_(update_solution,
		      UPDATE_SOLUTION)(long* Mesh_pointer,long* CG_pointer, int* myrank, realw* dt_in  )
{
	realw dt ;
	dt = *dt_in;
	Mesh* mp = (Mesh*)(*Mesh_pointer);
	printf("\nMYRANK:%d\n",*myrank);
	conjugate_gradient *newCG = (conjugate_gradient*)(*CG_pointer);

	std::cout << "\nvalue of deltat:" << dt << std::endl;
	printf("value of deltat : %f\n", dt);
//	cuProfilerInitialize();
	double start = get_time();

	cudaProfilerStart();
	
/*	printf("sizeof int:%d",sizeof(int));
	printf("sizeof int:%d",sizeof(int));


	printf("sizeof float:%d",sizeof(float));*/
	print_CUDA_error_if_any(cudaMemcpy(newCG->d_load, mp->d_accel, mp->NGLOB_AB*3*sizeof(realw), cudaMemcpyDeviceToDevice),100001);
	print_CUDA_error_if_any(cudaMemcpy(newCG->d_MASKAX, mp->d_maskax, mp->NGLOB_AB*3*sizeof(realw), cudaMemcpyDeviceToDevice),10002);
	print_CUDA_error_if_any(cudaMemcpy(newCG->d_MASKX, mp->d_maskax, mp->NGLOB_AB*3*sizeof(realw), cudaMemcpyDeviceToDevice),10003);

	newCG->reinit(2.0/(dt*dt));
//	newCG->get_field_from_gpu();


	for(int i = 0 ; i < 1 ; i++)
	{
		newCG->reinit(2.0/(dt*dt));

/*		for(int j = 0; j < 10000; j++)
		{
			newCG->update_val_dire();

		}*/
		newCG->solve(1e-10);
	}


	cudaProfilerStop();
		//newCG->compute_forces_call()
//		newCG->update_val_dire();
//	newCG->get_field_from_gpu();
	cudaMemcpy(mp->d_accel, mp->d_solution, mp->NGLOB_AB*3*sizeof(realw),cudaMemcpyDeviceToDevice);
	printf("\n time elapsed: %lf ",get_time()-start);
//	abort();
}

extern "C"
void FC_FUNC_(compute_force_on_fault,
              COMPUTE_FORCE_ON_FAULT)(long* Mesh_pointer,
                                long* Fault_pointer,
                                int* myrank,
								int* it)
{
	Mesh *mp = (Mesh*)(*Mesh_pointer);
	cudaMemset(mp->d_accel,0, mp->NGLOB_AB*3*sizeof(realw));
	compute_fault_gpu_(mp->d_accel,mp->d_displ,mp->d_maskax, mp->d_maskax );

}


