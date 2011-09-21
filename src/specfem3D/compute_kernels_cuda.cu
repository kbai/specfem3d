#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "mesh_constants_cuda.h"
#define MAX(x,y)                    (((x) < (y)) ? (y) : (x))
void print_CUDA_error_if_any(cudaError_t err, int num);

#define ENABLE_VERY_SLOW_ERROR_CHECKING

#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + (y)*xsize + (z)*xsize*ysize
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize
#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize + (j)*xsize*ysize*zsize*isize


__global__ void compute_kernels_cudakernel(int* ispec_is_elastic, int* ibool,
					   float* accel,
					   float* b_displ,
					   float* epsilondev_xx,  
					   float* epsilondev_yy,  
					   float* epsilondev_xy,  
					   float* epsilondev_xz,  
					   float* epsilondev_yz,  
					   float* b_epsilondev_xx,
					   float* b_epsilondev_yy,
					   float* b_epsilondev_xy,
					   float* b_epsilondev_xz,
					   float* b_epsilondev_yz,
					   float* rho_kl,					   
					   float deltat,
					   float* mu_kl,
					   float* kappa_kl,
					   float* epsilon_trace_over_3,
					   float* b_epsilon_trace_over_3,
					   int NSPEC_AB,
					   float* d_debug) {

  int ispec = blockIdx.x + blockIdx.y*gridDim.x;
  if(ispec<NSPEC_AB) { // handles case when there is 1 extra block (due to rectangular grid)
    int ijk = threadIdx.x;
    int ijk_ispec = ijk + 125*ispec;
    int iglob = ibool[ijk_ispec]-1;

    // if(ispec_is_elastic[ispec]) { // leave out until have acoustic coupling
    if(1) {
      
      
      if(ijk_ispec == 9480531) {
      	d_debug[0] = rho_kl[ijk_ispec];
      	d_debug[1] = accel[3*iglob];
      	d_debug[2] = b_displ[3*iglob];
	d_debug[3] = deltat * (accel[3*iglob]*b_displ[3*iglob]+
      				     accel[3*iglob+1]*b_displ[3*iglob+1]+
      				     accel[3*iglob+2]*b_displ[3*iglob+2]);
      }
      
      rho_kl[ijk_ispec] += deltat * (accel[3*iglob]*b_displ[3*iglob]+
      				     accel[3*iglob+1]*b_displ[3*iglob+1]+
      				     accel[3*iglob+2]*b_displ[3*iglob+2]);

      
      
      // if(rho_kl[ijk_ispec] < 1.9983e+18) {
      // atomicAdd(&d_debug[3],1.0);
      // d_debug[4] = ijk_ispec;
	// d_debug[0] = rho_kl[ijk_ispec];
	// d_debug[1] = accel[3*iglob];
	// d_debug[2] = b_displ[3*iglob];
      // }
      
      mu_kl[ijk_ispec] += deltat * (epsilondev_xx[ijk_ispec]*b_epsilondev_xx[ijk_ispec]+ // 1*b1
				    epsilondev_yy[ijk_ispec]*b_epsilondev_yy[ijk_ispec]+ // 2*b2
				    (epsilondev_xx[ijk_ispec]+epsilondev_yy[ijk_ispec])*
				    (b_epsilondev_xx[ijk_ispec]+b_epsilondev_yy[ijk_ispec])+
				    2*(epsilondev_xy[ijk_ispec]*b_epsilondev_xy[ijk_ispec]+
				       epsilondev_xz[ijk_ispec]*b_epsilondev_xz[ijk_ispec]+
				       epsilondev_yz[ijk_ispec]*b_epsilondev_yz[ijk_ispec]));
      
      kappa_kl[ijk_ispec] += deltat*(9*epsilon_trace_over_3[ijk_ispec]*
				     b_epsilon_trace_over_3[ijk_ispec]);
    
    }
  }
}
					   
					   

extern "C" void compute_kernels_cuda_(long* Mesh_pointer, int* NOISE_TOMOGRAPHY,
				     int* ELASTIC_SIMULATION, int* SAVE_MOHO_MESH,float* deltat) {

  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

  int blocksize = 125; // NGLLX*NGLLY*NGLLZ
  int num_blocks_x = mp->NSPEC_AB;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }
  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(blocksize,1,1);
  
  float* d_debug;
  float* h_debug;
  h_debug = (float*)calloc(128,sizeof(float));
  cudaMalloc((void**)&d_debug,128*sizeof(float));
  cudaMemcpy(d_debug,h_debug,128*sizeof(float),cudaMemcpyHostToDevice);
  
  
  compute_kernels_cudakernel<<<grid,threads>>>(mp->d_ispec_is_elastic,mp->d_ibool,
					       mp->d_accel, mp->d_b_displ,
					       mp->d_epsilondev_xx,
					       mp->d_epsilondev_yy,
					       mp->d_epsilondev_xy,
					       mp->d_epsilondev_xz,
					       mp->d_epsilondev_yz,
					       mp->d_b_epsilondev_xx,
					       mp->d_b_epsilondev_yy,
					       mp->d_b_epsilondev_xy,
					       mp->d_b_epsilondev_xz,
					       mp->d_b_epsilondev_yz,
					       mp->d_rho_kl,
					       *deltat,
					       mp->d_mu_kl,
					       mp->d_kappa_kl,
					       mp->d_epsilon_trace_over_3,
					       mp->d_b_epsilon_trace_over_3,
					       mp->NSPEC_AB,
					       d_debug);

  cudaMemcpy(h_debug,d_debug,128*sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(d_debug);
  // for(int i=0;i<5;i++) {
  // printf("d_debug[%d]=%e\n",i,h_debug[i]);
  // }
  free(h_debug);
  
  // float* h_rho = (float*)malloc(sizeof(float)*mp->NSPEC_AB*125);
  // float maxval = 0;
  // cudaMemcpy(h_rho,mp->d_rho_kl,sizeof(float)*mp->NSPEC_AB*125,cudaMemcpyDeviceToHost);
  // int number_big_values = 0;
  // for(int i=0;i<mp->NSPEC_AB*125;i++) {
  // maxval = MAX(maxval,fabsf(h_rho[i]));
  // if(fabsf(h_rho[i]) > 1e10) {
  // number_big_values++;
  // }
  // }
  
  // printf("maval rho = %e, number>1e10 = %d vs. %d\n",maxval,number_big_values,mp->NSPEC_AB*125);
  
  #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  // sync and check to catch errors from previous async operations
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr,"Error launching/running compute_kernels_cudakernel: %s\n", cudaGetErrorString(err));
      exit(1);
    }
#endif
  
  
}

extern "C" void transfer_sensitivity_kernels_to_host_(long* Mesh_pointer, float* h_rho_kl,
						      float* h_mu_kl, float* h_kappa_kl,
						      float* h_Sigma_kl,int* NSPEC_AB,int* NSPEC_AB_VAL) {

  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

  print_CUDA_error_if_any(cudaMemcpy(h_rho_kl,mp->d_rho_kl,*NSPEC_AB*125*sizeof(float),
				     cudaMemcpyDeviceToHost),1);
  print_CUDA_error_if_any(cudaMemcpy(h_mu_kl,mp->d_mu_kl,*NSPEC_AB*125*sizeof(float),
				     cudaMemcpyDeviceToHost),1);
  print_CUDA_error_if_any(cudaMemcpy(h_kappa_kl,mp->d_kappa_kl,*NSPEC_AB*125*sizeof(float),
				     cudaMemcpyDeviceToHost),1);
  print_CUDA_error_if_any(cudaMemcpy(h_Sigma_kl,mp->d_Sigma_kl,125*(*NSPEC_AB_VAL)*sizeof(float),
				     cudaMemcpyHostToDevice),4);
  
}

__global__ void compute_kernels_strength_noise_cuda_kernel(float* displ, int* free_surface_ispec,int* free_surface_ijk, int* ibool, float* noise_surface_movie, float* normal_x_noise, float* normal_y_noise, float* normal_z_noise, float* Sigma_kl, float deltat,int num_free_surface_faces, float* d_debug) {
  int iface = blockIdx.x + blockIdx.y*gridDim.x;
  if(iface<num_free_surface_faces) {

    int ispec = free_surface_ispec[iface]-1;
    int igll = threadIdx.x;        
    int ipoin = igll + 25*iface;
    int i = free_surface_ijk[INDEX3(3,25,0,igll,iface)]-1;
    int j = free_surface_ijk[INDEX3(3,25,0,igll,iface)]-1;
    int k = free_surface_ijk[INDEX3(3,25,0,igll,iface)]-1;
    
    int iglob = ibool[INDEX4(5,5,5,i,j,k,ispec)]-1;
    
    float eta = (noise_surface_movie[INDEX3(3,25,0,igll,iface)]*normal_x_noise[ipoin]+
		 noise_surface_movie[INDEX3(3,25,1,igll,iface)]*normal_y_noise[ipoin]+ 
		 noise_surface_movie[INDEX3(3,25,2,igll,iface)]*normal_z_noise[ipoin]);

    // if(ijk_ispec == 78496) {
    //   d_debug[0] = Sigma_kl[ijk_ispec];
    //   d_debug[1] = eta;
    //   d_debug[2] = normal_x_noise[ipoin];
    //   d_debug[3] = normal_y_noise[ipoin];
    //   d_debug[4] = normal_z_noise[ipoin];
    //   d_debug[5] = displ[3*iglob+2];      
    //   d_debug[6] = deltat*eta*normal_z_noise[ipoin]*displ[2+3*iglob];
    //   d_debug[7] = 0.008*1.000000e-24*normal_z_noise[ipoin]*3.740546e-13;
    // }
    
    Sigma_kl[INDEX4(5,5,5,i,j,k,ispec)] += deltat*eta*(normal_x_noise[ipoin]*displ[3*iglob]+
				       normal_y_noise[ipoin]*displ[1+3*iglob]+
				       normal_z_noise[ipoin]*displ[2+3*iglob]);
  }
  
  
}

extern "C" void compute_kernels_strength_noise_cuda_(long* Mesh_pointer, float* h_noise_surface_movie,
						     int* num_free_surface_faces_f,int* deltat) {
  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  int num_free_surface_faces = *num_free_surface_faces_f;

  cudaMemcpy(mp->d_noise_surface_movie,h_noise_surface_movie,3*25*num_free_surface_faces*sizeof(float),cudaMemcpyHostToDevice);


  int num_blocks_x = num_free_surface_faces;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(25,1,1);

  // float* h_debug = (float*)calloc(128,sizeof(float));
  float* d_debug;
  // cudaMalloc((void**)&d_debug,128*sizeof(float));
  // cudaMemcpy(d_debug,h_debug,128*sizeof(float),cudaMemcpyHostToDevice);
  
  compute_kernels_strength_noise_cuda_kernel<<<grid,threads>>>(mp->d_displ,
							       mp->d_free_surface_ispec,
							       mp->d_free_surface_ijk,
							       mp->d_ibool,
							       mp->d_noise_surface_movie,
							       mp->d_normal_x_noise,
							       mp->d_normal_y_noise,
							       mp->d_normal_z_noise,
							       mp->d_Sigma_kl,*deltat,
							       num_free_surface_faces,
							       d_debug);

  // cudaMemcpy(h_debug,d_debug,128*sizeof(float),cudaMemcpyDeviceToHost);
  // for(int i=0;i<8;i++) {
  //   printf("debug[%d]= %e\n",i,h_debug[i]);
  // }
  
  #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("compute_kernels_strength_noise_cuda_kernel");
  #endif
  
}
