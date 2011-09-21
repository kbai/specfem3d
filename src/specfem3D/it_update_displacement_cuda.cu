#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>

#include "mesh_constants_cuda.h"

#define ENABLE_VERY_SLOW_ERROR_CHECKING


// typedef struct mesh_ {
  
//   int NGLLX; int NSPEC_AB;
//   int NGLOB_AB;
//   float* d_xix; float* d_xiy; float* d_xiz;
//   float* d_etax; float* d_etay; float* d_etaz;
//   float* d_gammax; float* d_gammay; float* d_gammaz;
//   float* d_kappav; float* d_muv;
//   int* d_ibool;
//   float* d_displ; float* d_veloc; float* d_accel;
//   int* d_phase_ispec_inner_elastic;
//   int d_num_phase_ispec_elastic;
//   float* d_rmass;
  
// } Mesh;

typedef float real;

  
  __global__ void UpdateDispVeloc_kernel(real* displ, real* veloc,
					 real* accel, int size,
					 real deltat, real deltatsqover2, real deltatover2) {
    int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;

    /* because of block and grid sizing problems, there is a small */
    /* amount of buffer at the end of the calculation */
    if(id < size) {     
      displ[id] = displ[id] + deltat*veloc[id] + deltatsqover2*accel[id];
      veloc[id] = veloc[id] + deltatover2*accel[id];
      accel[id] = 0; // can do this using memset...not sure if faster
    }
  }
#define CUBLAS_ERROR(s,n)  if (s != CUBLAS_STATUS_SUCCESS) {	\
    fprintf (stderr, "CUBLAS Memory Write Error @ %d\n",n);	\
    exit(EXIT_FAILURE); }


extern "C" void it_update_displacement_scheme_cuda_(long* Mesh_pointer_f,int* size_F, float* deltat_F, float* deltatsqover2_F, float* deltatover2_F,int* SIMULATION_TYPE, float* b_deltat_F, float* b_deltatsqover2_F, float* b_deltatover2_F) {

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper
  
  int i,device;

  int size = *size_F;
  real deltat = *deltat_F;
  real deltatsqover2 = *deltatsqover2_F;
  real deltatover2 = *deltatover2_F;
  real b_deltat = *b_deltat_F;
  real b_deltatsqover2 = *b_deltatsqover2_F;
  real b_deltatover2 = *b_deltatover2_F;
  cublasStatus status;
  
  int blocksize = 128;
  int size_padded = ((int)ceil(((double)size)/((double)blocksize)))*blocksize;  
  
  int num_blocks_x = size_padded/blocksize;  
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }
  
  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(blocksize,1,1);

  
  exit_on_cuda_error("Before UpdateDispVeloc_kernel");

  //launch kernel
  UpdateDispVeloc_kernel<<<grid,threads>>>(mp->d_displ,mp->d_veloc,mp->d_accel,
					   size,deltat,deltatsqover2,deltatover2);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  //printf("checking updatedispl_kernel launch...with %dx%d blocks\n",num_blocks_x,num_blocks_y);
  // sync and check to catch errors from previous async operations
  exit_on_cuda_error("UpdateDispVeloc_kernel");
#endif


  // kernel for backward fields
  if(*SIMULATION_TYPE == 3) {
    
    UpdateDispVeloc_kernel<<<grid,threads>>>(mp->d_b_displ,mp->d_b_veloc,mp->d_b_accel,
					     size,b_deltat, b_deltatsqover2, b_deltatover2);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
    //printf("checking updatedispl_kernel launch...with %dx%d blocks\n",num_blocks_x,num_blocks_y);
    // sync and check to catch errors from previous async operations
    cudaThreadSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      {
	fprintf(stderr,"Error after SIM_TYPE==3 UpdateDispVeloc_kernel: %s\n", cudaGetErrorString(err));
	exit(1);
      }
#endif

  }
  
  cudaThreadSynchronize();
  
}

