/*
 !=====================================================================
 !
 !               S p e c f e m 3 D  V e r s i o n  2 . 0
 !               ---------------------------------------
 !
 !          Main authors: Dimitri Komatitsch and Jeroen Tromp
 !    Princeton University, USA and University of Pau / CNRS / INRIA
 ! (c) Princeton University / California Institute of Technology and University of Pau / CNRS / INRIA
 !                            April 2011
 !
 ! This program is free software; you can redistribute it and/or modify
 ! it under the terms of the GNU General Public License as published by
 ! the Free Software Foundation; either version 2 of the License, or
 ! (at your option) any later version.
 !
 ! This program is distributed in the hope that it will be useful,
 ! but WITHOUT ANY WARRANTY; without even the implied warranty of
 ! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 ! GNU General Public License for more details.
 !
 ! You should have received a copy of the GNU General Public License along
 ! with this program; if not, write to the Free Software Foundation, Inc.,
 ! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 !
 !=====================================================================
 */

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "config.h"
#include "mesh_constants_cuda.h"


/* ----------------------------------------------------------------------------------------------- */

// ELASTIC SIMULATIONS

/* ----------------------------------------------------------------------------------------------- */

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

  // handles case when there is 1 extra block (due to rectangular grid)
  if(ispec < NSPEC_AB) { 

    // elastic elements only
    if(ispec_is_elastic[ispec] == 1) { 
  
      int ijk = threadIdx.x;
      int ijk_ispec = ijk + 125*ispec;
      int iglob = ibool[ijk_ispec] - 1 ;

      // debug      
//      if(ijk_ispec == 9480531) {
//      	d_debug[0] = rho_kl[ijk_ispec];
//      	d_debug[1] = accel[3*iglob];
//      	d_debug[2] = b_displ[3*iglob];
//        d_debug[3] = deltat * (accel[3*iglob]*b_displ[3*iglob]+
//                               accel[3*iglob+1]*b_displ[3*iglob+1]+
//                               accel[3*iglob+2]*b_displ[3*iglob+2]);
//      }
      
      
      // isotropic kernels:      
      // density kernel
      rho_kl[ijk_ispec] += deltat * (accel[3*iglob]*b_displ[3*iglob]+
                                     accel[3*iglob+1]*b_displ[3*iglob+1]+
                                     accel[3*iglob+2]*b_displ[3*iglob+2]);

      
      // debug
      // if(rho_kl[ijk_ispec] < 1.9983e+18) {
      // atomicAdd(&d_debug[3],1.0);
      // d_debug[4] = ijk_ispec;
      // d_debug[0] = rho_kl[ijk_ispec];
      // d_debug[1] = accel[3*iglob];
      // d_debug[2] = b_displ[3*iglob];
      // }
      
      // shear modulus kernel
      mu_kl[ijk_ispec] += deltat * (epsilondev_xx[ijk_ispec]*b_epsilondev_xx[ijk_ispec]+ // 1*b1
                                    epsilondev_yy[ijk_ispec]*b_epsilondev_yy[ijk_ispec]+ // 2*b2
                                    (epsilondev_xx[ijk_ispec]+epsilondev_yy[ijk_ispec])*
                                    (b_epsilondev_xx[ijk_ispec]+b_epsilondev_yy[ijk_ispec])+
                                    2*(epsilondev_xy[ijk_ispec]*b_epsilondev_xy[ijk_ispec]+
                                       epsilondev_xz[ijk_ispec]*b_epsilondev_xz[ijk_ispec]+
                                       epsilondev_yz[ijk_ispec]*b_epsilondev_yz[ijk_ispec]));
      
      // bulk modulus kernel
      kappa_kl[ijk_ispec] += deltat*(9*epsilon_trace_over_3[ijk_ispec]*
                                     b_epsilon_trace_over_3[ijk_ispec]);
    
    }
  }
}
					   
/* ----------------------------------------------------------------------------------------------- */					   

extern "C" 
void FC_FUNC_(compute_kernels_elastic_cuda,
              COMPUTE_KERNELS_ELASTIC_CUDA)(long* Mesh_pointer,
                                            float* deltat) {
TRACE("compute_kernels_elastic_cuda");

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
  /*
  float* h_debug;
  h_debug = (float*)calloc(128,sizeof(float));
  cudaMalloc((void**)&d_debug,128*sizeof(float));
  cudaMemcpy(d_debug,h_debug,128*sizeof(float),cudaMemcpyHostToDevice);
  */
  
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
  /*
  cudaMemcpy(h_debug,d_debug,128*sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(d_debug);
  */
  // for(int i=0;i<5;i++) {
  // printf("d_debug[%d]=%e\n",i,h_debug[i]);
  // }
  /*
  free(h_debug);
  */
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
  exit_on_cuda_error("compute_kernels_elastic_cuda");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

// NOISE SIMULATIONS

/* ----------------------------------------------------------------------------------------------- */


__global__ void compute_kernels_strength_noise_cuda_kernel(float* displ, 
                                                           int* free_surface_ispec,
                                                           int* free_surface_ijk,
                                                           int* ibool, 
                                                           float* noise_surface_movie, 
                                                           float* normal_x_noise, 
                                                           float* normal_y_noise, 
                                                           float* normal_z_noise, 
                                                           float* Sigma_kl, 
                                                           float deltat,
                                                           int num_free_surface_faces, 
                                                           float* d_debug) {
  int iface = blockIdx.x + blockIdx.y*gridDim.x;
  
  if(iface < num_free_surface_faces) {

    int ispec = free_surface_ispec[iface]-1;
    int igll = threadIdx.x;        
    int ipoin = igll + 25*iface;
    int i = free_surface_ijk[INDEX3(NDIM,NGLL2,0,igll,iface)] - 1 ;
    int j = free_surface_ijk[INDEX3(NDIM,NGLL2,1,igll,iface)] - 1;
    int k = free_surface_ijk[INDEX3(NDIM,NGLL2,2,igll,iface)] - 1;
    
    int iglob = ibool[INDEX4(5,5,5,i,j,k,ispec)] - 1 ;
    
    float eta = ( noise_surface_movie[INDEX3(NDIM,NGLL2,0,igll,iface)]*normal_x_noise[ipoin]+
                 noise_surface_movie[INDEX3(NDIM,NGLL2,1,igll,iface)]*normal_y_noise[ipoin]+ 
                 noise_surface_movie[INDEX3(NDIM,NGLL2,2,igll,iface)]*normal_z_noise[ipoin]);

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

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(compute_kernels_strength_noise_cuda,
              COMPUTE_KERNELS_STRENGTH_NOISE_CUDA)(long* Mesh_pointer, 
                                                    float* h_noise_surface_movie,
                                                    int* num_free_surface_faces_f,
                                                    float* deltat) {

TRACE("compute_kernels_strength_noise_cuda");
                                                    
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



/* ----------------------------------------------------------------------------------------------- */

// ACOUSTIC SIMULATIONS

/* ----------------------------------------------------------------------------------------------- */


__device__ void compute_gradient_kernel(int ijk,
                                        int ispec,
                                        float* scalar_field,
                                        float* vector_field_element,
                                        float* hprime_xx,
                                        float* hprime_yy,
                                        float* hprime_zz,
                                        float* d_xix, 
                                        float* d_xiy, 
                                        float* d_xiz, 
                                        float* d_etax, 
                                        float* d_etay, 
                                        float* d_etaz, 
                                        float* d_gammax, 
                                        float* d_gammay, 
                                        float* d_gammaz,
                                        float rhol) {
  
  float temp1l,temp2l,temp3l;
  float hp1,hp2,hp3;
  float xixl,xiyl,xizl,etaxl,etayl,etazl,gammaxl,gammayl,gammazl;
  float rho_invl;
  int l,offset,offset1,offset2,offset3;
  
  //const int NGLLX = 5;
  const int NGLL3_ALIGN = 128;
  
  int K = (ijk/NGLL2);
  int J = ((ijk-K*NGLL2)/NGLLX);
  int I = (ijk-K*NGLL2-J*NGLLX);
  
  // derivative along x
  temp1l = 0.f;
  for( l=0; l<NGLLX;l++){
    hp1 = hprime_xx[l*NGLLX+I];
    offset1 = K*NGLL2+J*NGLLX+l;
    temp1l += scalar_field[offset1]*hp1;
  }
  
  // derivative along y
  temp2l = 0.f;
  for( l=0; l<NGLLX;l++){
    hp2 = hprime_yy[l*NGLLX+J];
    offset2 = K*NGLL2+l*NGLLX+I;
    temp2l += scalar_field[offset2]*hp2;
  }
  
  // derivative along z    
  temp3l = 0.f;
  for( l=0; l<NGLLX;l++){
    hp3 = hprime_zz[l*NGLLX+K];
    offset3 = l*NGLL2+J*NGLLX+I;
    temp3l += scalar_field[offset3]*hp3;
    
  }
  
  offset = ispec*NGLL3_ALIGN + ijk;
  
  xixl = d_xix[offset];
  xiyl = d_xiy[offset];
  xizl = d_xiz[offset];
  etaxl = d_etax[offset];
  etayl = d_etay[offset];
  etazl = d_etaz[offset];
  gammaxl = d_gammax[offset];
  gammayl = d_gammay[offset];
  gammazl = d_gammaz[offset];
  
  rho_invl = 1.0f / rhol;
  
  // derivatives of acoustic scalar potential field on GLL points
  vector_field_element[0] = (temp1l*xixl + temp2l*etaxl + temp3l*gammaxl) * rho_invl;
  vector_field_element[1] = (temp1l*xiyl + temp2l*etayl + temp3l*gammayl) * rho_invl;
  vector_field_element[2] = (temp1l*xizl + temp2l*etazl + temp3l*gammazl) * rho_invl;  
  
}

/* ----------------------------------------------------------------------------------------------- */


__global__ void compute_kernels_acoustic_kernel(int* ispec_is_acoustic, 
                                                int* ibool,
                                                float* rhostore,
                                                float* kappastore,
                                                float* hprime_xx,
                                                float* hprime_yy,
                                                float* hprime_zz,
                                                float* d_xix, 
                                                float* d_xiy, 
                                                float* d_xiz, 
                                                float* d_etax, 
                                                float* d_etay, 
                                                float* d_etaz, 
                                                float* d_gammax, 
                                                float* d_gammay, 
                                                float* d_gammaz,                                                
                                                float* potential_dot_dot_acoustic,
                                                float* b_potential_acoustic,
                                                float* b_potential_dot_dot_acoustic,
                                                float* rho_ac_kl,					   
                                                float* kappa_ac_kl,
                                                float deltat,
                                                int NSPEC_AB) {
  
  int ispec = blockIdx.x + blockIdx.y*gridDim.x;

  // handles case when there is 1 extra block (due to rectangular grid)    
  if( ispec < NSPEC_AB ){
  
    // acoustic elements only
    if( ispec_is_acoustic[ispec] == 1) { 
    
      int ijk = threadIdx.x;
      
      // local and global indices
      int ijk_ispec = ijk + 125*ispec;
      int ijk_ispec_padded = ijk + 128*ispec;
      int iglob = ibool[ijk_ispec] - 1;
      
      float accel_elm[3];
      float b_displ_elm[3];
      float rhol,kappal;
      
      // shared memory between all threads within this block
      __shared__ float scalar_field_displ[125];    
      __shared__ float scalar_field_accel[125];          
      
      // copy field values
      scalar_field_displ[ijk] = b_potential_acoustic[iglob];
      scalar_field_accel[ijk] = potential_dot_dot_acoustic[iglob];
      __syncthreads();
      
      // gets material parameter
      rhol = rhostore[ijk_ispec_padded];
      
      // displacement vector from backward field
      compute_gradient_kernel(ijk,ispec,scalar_field_displ,b_displ_elm,
                              hprime_xx,hprime_yy,hprime_zz,
                              d_xix,d_xiy,d_xiz,d_etax,d_etay,d_etaz,d_gammax,d_gammay,d_gammaz,
                              rhol);
      
      // acceleration vector
      compute_gradient_kernel(ijk,ispec,scalar_field_accel,accel_elm,
                              hprime_xx,hprime_yy,hprime_zz,
                              d_xix,d_xiy,d_xiz,d_etax,d_etay,d_etaz,d_gammax,d_gammay,d_gammaz,
                              rhol);
      
      // density kernel
      rho_ac_kl[ijk_ispec] -= deltat * rhol * (accel_elm[0]*b_displ_elm[0] +
                                               accel_elm[1]*b_displ_elm[1] +
                                               accel_elm[2]*b_displ_elm[2]);
      
      // bulk modulus kernel
      kappal = kappastore[ijk_ispec];
      kappa_ac_kl[ijk_ispec] -= deltat / kappal * potential_dot_dot_acoustic[iglob] 
                                                * b_potential_dot_dot_acoustic[iglob];    
    }
  }  
}

/* ----------------------------------------------------------------------------------------------- */


extern "C" 
void FC_FUNC_(compute_kernels_acoustic_cuda,
              COMPUTE_KERNELS_ACOUSTIC_CUDA)(
                                             long* Mesh_pointer, 
                                             float* deltat) {
  
TRACE("compute_kernels_acoustic_cuda");
  
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
  
  compute_kernels_acoustic_kernel<<<grid,threads>>>(mp->d_ispec_is_acoustic,
                                                    mp->d_ibool,
                                                    mp->d_rhostore,
                                                    mp->d_kappastore,
                                                    mp->d_hprime_xx,
                                                    mp->d_hprime_yy,
                                                    mp->d_hprime_zz,                                                    
                                                    mp->d_xix, 
                                                    mp->d_xiy, 
                                                    mp->d_xiz,
                                                    mp->d_etax, 
                                                    mp->d_etay, 
                                                    mp->d_etaz,
                                                    mp->d_gammax, 
                                                    mp->d_gammay, 
                                                    mp->d_gammaz,                                                    
                                                    mp->d_potential_dot_dot_acoustic, 
                                                    mp->d_b_potential_acoustic,
                                                    mp->d_b_potential_dot_dot_acoustic,
                                                    mp->d_rho_ac_kl,
                                                    mp->d_kappa_ac_kl,
                                                    *deltat,
                                                    mp->NSPEC_AB);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("compute_kernels_acoustic_kernel");
#endif
}

