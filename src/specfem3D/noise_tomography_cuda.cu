#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>

#include "mesh_constants_cuda.h"

// #include "epik_user.h"

#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + (y)*xsize + (z)*xsize*ysize
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize
#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize + (j)*xsize*ysize*zsize*isize

typedef float real;

#define ENABLE_VERY_SLOW_ERROR_CHECKING

__global__ void transfer_surface_to_host_kernel(int* free_surface_ispec,int* free_surface_ijk, int num_free_surface_faces, int* ibool, real* displ, real* noise_surface_movie) {
  int igll = threadIdx.x;
  int iface = blockIdx.x + blockIdx.y*gridDim.x;
  
  // int id = tx + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
  
  if(iface < num_free_surface_faces) {
    int ispec = free_surface_ispec[iface]-1; //-1 for C-based indexing

    int i = free_surface_ijk[0+3*(igll + 25*(iface))]-1;
    int j = free_surface_ijk[1+3*(igll + 25*(iface))]-1;
    int k = free_surface_ijk[2+3*(igll + 25*(iface))]-1;
        
    int iglob = ibool[INDEX4(5,5,5,i,j,k,ispec)]-1;    
    
    noise_surface_movie[INDEX3(3,25,0,igll,iface)] = displ[iglob*3];
    noise_surface_movie[INDEX3(3,25,1,igll,iface)] = displ[iglob*3+1];
    noise_surface_movie[INDEX3(3,25,2,igll,iface)] = displ[iglob*3+2];
  }
}

extern "C" void fortranflush_(int* rank)
{
  
  fflush(stdout);
  fflush(stderr);
  printf("Flushing proc %d!\n",*rank);
}

extern "C" void fortranprint_(int* id) {
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  printf("%d: sends msg_id %d\n",procid,*id);
}

extern "C" void fortranprintf_(float* val) {
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  printf("%d: sends val %e\n",procid,*val);
}

extern "C" void fortranprintd_(double* val) {
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  printf("%d: sends val %e\n",procid,*val);
}

// randomize displ for testing
extern "C" void make_displ_rand_(long* Mesh_pointer_f,float* h_displ) {

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper  
  // float* displ_rnd = (float*)malloc(mp->NGLOB_AB*3*sizeof(float));
  for(int i=0;i<mp->NGLOB_AB*3;i++) {
    h_displ[i] = rand();
  }
  cudaMemcpy(mp->d_displ,h_displ,mp->NGLOB_AB*3*sizeof(float),cudaMemcpyHostToDevice);
}

extern "C" void transfer_surface_to_host_(long* Mesh_pointer_f,real* h_noise_surface_movie,int* num_free_surface_faces) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper  
  int num_blocks_x = *num_free_surface_faces;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }
  dim3 grid(num_blocks_x,num_blocks_y,1);
  dim3 threads(25,1,1);  
  
  transfer_surface_to_host_kernel<<<grid,threads>>>(mp->d_free_surface_ispec,mp->d_free_surface_ijk, *num_free_surface_faces, mp->d_ibool, mp->d_displ, mp->d_noise_surface_movie);  

  cudaMemcpy(h_noise_surface_movie,mp->d_noise_surface_movie,3*25*(*num_free_surface_faces)*sizeof(real),cudaMemcpyDeviceToHost);  

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  // sync and check to catch errors from previous async operations
  cudaThreadSynchronize();
  exit_on_cuda_error("transfer_surface_to_host");
#endif
  
}



__global__ void noise_read_add_surface_movie_cuda_kernel(real* accel, int* ibool, int* free_surface_ispec,int* free_surface_ijk, int num_free_surface_faces, real* noise_surface_movie, real* normal_x_noise, real* normal_y_noise, real* normal_z_noise, real* mask_noise, real* free_surface_jacobian2Dw, real* wgllwgll_xy,float* d_debug) {

  int iface = blockIdx.x + gridDim.x*blockIdx.y; // surface element id

  // when nspec_top > 65535, but mod(nspec_top,2) > 0, we end up with an extra block.
  if(iface < num_free_surface_faces) { 
    int ispec = free_surface_ispec[iface]-1;
    
    int igll = threadIdx.x;
    
    int ipoin = 25*iface + igll;
    int i=free_surface_ijk[0+3*(igll + 25*(iface))]-1;
    int j=free_surface_ijk[1+3*(igll + 25*(iface))]-1;
    int k=free_surface_ijk[2+3*(igll + 25*(iface))]-1;    
    
    int iglob = ibool[INDEX4(5,5,5,i,j,k,ispec)]-1;
    
    real normal_x = normal_x_noise[ipoin];
    real normal_y = normal_y_noise[ipoin];
    real normal_z = normal_z_noise[ipoin];

    real eta = (noise_surface_movie[INDEX3(3,25,0,igll,iface)]*normal_x + 
		noise_surface_movie[INDEX3(3,25,1,igll,iface)]*normal_y +
		noise_surface_movie[INDEX3(3,25,2,igll,iface)]*normal_z);
    
    // error from cuda-memcheck and ddt seems "incorrect", because we
    // are passing a __constant__ variable pointer around like it was
    // made using cudaMalloc, which *may* be "incorrect", but produces
    // correct results.
    
    // ========= Invalid __global__ read of size
    // 4 ========= at 0x00000cd8 in
    // compute_add_sources_cuda.cu:260:noise_read_add_surface_movie_cuda_kernel
    // ========= by thread (0,0,0) in block (3443,0) ========= Address
    // 0x203000c8 is out of bounds
    
    // non atomic version for speed testing -- atomic updates are needed for correctness
    // accel[3*iglob] +=   eta*mask_noise[ipoin] * normal_x * wgllwgll_xy[tx] * free_surface_jacobian2Dw[tx + 25*ispec2D];
    // accel[3*iglob+1] += eta*mask_noise[ipoin] * normal_y * wgllwgll_xy[tx] * free_surface_jacobian2Dw[tx + 25*ispec2D];
    // accel[3*iglob+2] += eta*mask_noise[ipoin] * normal_z * wgllwgll_xy[tx] * free_surface_jacobian2Dw[tx + 25*ispec2D];

    // Fortran version in SVN -- note deletion of wgllwgll_xy?
    // accel(1,iglob) = accel(1,iglob) + eta * mask_noise(ipoin) * normal_x_noise(ipoin) &
    // * free_surface_jacobian2Dw(igll,iface) 
    // accel(2,iglob) = accel(2,iglob) + eta * mask_noise(ipoin) * normal_y_noise(ipoin) &
    // * free_surface_jacobian2Dw(igll,iface)
    // accel(3,iglob) = accel(3,iglob) + eta * mask_noise(ipoin) * normal_z_noise(ipoin) &
    // * free_surface_jacobian2Dw(igll,iface) ! wgllwgll_xy(i,j) * jacobian2D_top(i,j,iface)
    
    // atomicAdd(&accel[iglob*3]  ,eta*mask_noise[ipoin]*normal_x*wgllwgll_xy[tx]*free_surface_jacobian2Dw[igll+25*iface]);
    // atomicAdd(&accel[iglob*3+1],eta*mask_noise[ipoin]*normal_y*wgllwgll_xy[tx]*free_surface_jacobian2Dw[igll+25*iface]);
    // atomicAdd(&accel[iglob*3+2],eta*mask_noise[ipoin]*normal_z*wgllwgll_xy[tx]*free_surface_jacobian2Dw[igll+25*iface]);
    
    atomicAdd(&accel[iglob*3]  ,eta*mask_noise[ipoin]*normal_x*free_surface_jacobian2Dw[igll+25*iface]);
    atomicAdd(&accel[iglob*3+1],eta*mask_noise[ipoin]*normal_y*free_surface_jacobian2Dw[igll+25*iface]);
    atomicAdd(&accel[iglob*3+2],eta*mask_noise[ipoin]*normal_z*free_surface_jacobian2Dw[igll+25*iface]);
    
  }
}

extern "C" void noise_read_add_surface_movie_cuda_(long* Mesh_pointer_f, real* h_noise_surface_movie, int* num_free_surface_faces_f,int* NOISE_TOMOGRAPHYf) {

  // EPIK_TRACER("noise_read_add_surface_movie_cuda");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  int num_free_surface_faces = *num_free_surface_faces_f;
  int NOISE_TOMOGRAPHY = *NOISE_TOMOGRAPHYf;
  float* d_noise_surface_movie;
  cudaMalloc((void**)&d_noise_surface_movie,3*25*num_free_surface_faces*sizeof(float));
  cudaMemcpy(d_noise_surface_movie, h_noise_surface_movie,3*25*num_free_surface_faces*sizeof(real),cudaMemcpyHostToDevice);

  int num_blocks_x = num_free_surface_faces;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }
  dim3 grid(num_blocks_x,num_blocks_y,1);
  dim3 threads(25,1,1);

  // float* h_debug = (float*)calloc(128,sizeof(float));
  float* d_debug;
  // cudaMalloc((void**)&d_debug,128*sizeof(float));
  // cudaMemcpy(d_debug,h_debug,128*sizeof(float),cudaMemcpyHostToDevice);
  
  if(NOISE_TOMOGRAPHY == 2) { // add surface source to forward field
    noise_read_add_surface_movie_cuda_kernel<<<grid,threads>>>(mp->d_accel,
							       mp->d_ibool,
							       mp->d_free_surface_ispec,
							       mp->d_free_surface_ijk,
							       num_free_surface_faces,
							       d_noise_surface_movie,
							       mp->d_normal_x_noise,
							       mp->d_normal_y_noise,
							       mp->d_normal_z_noise,
							       mp->d_mask_noise,
							       mp->d_free_surface_jacobian2Dw,
							       mp->d_wgllwgll_xy,
							       d_debug);
  }
  else if(NOISE_TOMOGRAPHY==3) { // add surface source to adjoint (backward) field
    noise_read_add_surface_movie_cuda_kernel<<<grid,threads>>>(mp->d_b_accel,
							       mp->d_ibool,
							       mp->d_free_surface_ispec,
							       mp->d_free_surface_ijk,
							       num_free_surface_faces,
							       d_noise_surface_movie,
							       mp->d_normal_x_noise,
							       mp->d_normal_y_noise,
							       mp->d_normal_z_noise,
							       mp->d_mask_noise,
							       mp->d_free_surface_jacobian2Dw,
							       mp->d_wgllwgll_xy,
							       d_debug);
  }
  

  // cudaMemcpy(h_debug,d_debug,128*sizeof(float),cudaMemcpyDeviceToHost);
  // for(int i=0;i<8;i++) {
  // printf("debug[%d]= %e\n",i,h_debug[i]);
  // }
  // MPI_Abort(MPI_COMM_WORLD,1);
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  
  exit_on_cuda_error("noise_read_add_surface_movie_cuda_kernel");
  // sync and check to catch errors from previous async operations
  // cudaThreadSynchronize();
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess)
  //   {
  //     fprintf(stderr,"Error launching/running noise_read_add_surface_movie_cuda_kernel: %s\n", cudaGetErrorString(err));
  //     exit(1);
  //   }
#endif

  cudaFree(d_noise_surface_movie);
}
