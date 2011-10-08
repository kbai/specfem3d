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

#include <sys/time.h>
#include <sys/resource.h>

#include "config.h"
#include "mesh_constants_cuda.h"
#include "prepare_constants_cuda.h"

/* ----------------------------------------------------------------------------------------------- */

// Transfer functions

/* ----------------------------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------------------------------- */

// for ELASTIC simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_fields_to_device,
              TRANSFER_B_FIELDS_TO_DEVICE)(int* size, float* b_displ, float* b_veloc, float* b_accel,
                                           long* Mesh_pointer_f) {

TRACE("transfer_b_fields_to_device_");               

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  cudaMemcpy(mp->d_b_displ,b_displ,sizeof(float)*(*size),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_veloc,b_veloc,sizeof(float)*(*size),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_accel,b_accel,sizeof(float)*(*size),cudaMemcpyHostToDevice);
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_fields_to_device,
              TRANSFER_FIELDS_TO_DEVICE)(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f) {

TRACE("transfer_fields_to_device_");
    
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container

  print_CUDA_error_if_any(cudaMemcpy(mp->d_displ,displ,sizeof(float)*(*size),cudaMemcpyHostToDevice),40003);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_veloc,veloc,sizeof(float)*(*size),cudaMemcpyHostToDevice),40004);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_accel,accel,sizeof(float)*(*size),cudaMemcpyHostToDevice),40005);
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_fields_from_device,
              TRANSFER_B_FIELDS_FROM_DEVICE)(int* size, float* b_displ, float* b_veloc, float* b_accel,long* Mesh_pointer_f) {

TRACE("transfer_b_fields_from_device_");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  cudaMemcpy(b_displ,mp->d_b_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_veloc,mp->d_b_veloc,sizeof(float)*(*size),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_accel,mp->d_b_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost);
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_fields_from_device,
              TRANSFER_FIELDS_FROM_DEVICE)(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f) {
  
TRACE("transfer_fields_from_device_");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40006);
  print_CUDA_error_if_any(cudaMemcpy(veloc,mp->d_veloc,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40007);
  print_CUDA_error_if_any(cudaMemcpy(accel,mp->d_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40008);
  
  // printf("Transfered Fields From Device\n");
  // int procid;
  // MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  // printf("Quick check of answer for p:%d in transfer_fields_from_device\n",procid);
  // for(int i=0;i<5;i++) {
  // printf("accel[%d]=%2.20e\n",i,accel[i]);
  // }
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_accel_to_device,
              TRNASFER_ACCEL_TO_DEVICE)(int* size, float* accel,long* Mesh_pointer_f) {

TRACE("transfer_accel_to_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_accel,accel,sizeof(float)*(*size),cudaMemcpyHostToDevice),40016);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_accel_from_device,
              TRANSFER_ACCEL_FROM_DEVICE)(int* size, float* accel,long* Mesh_pointer_f) {

TRACE("transfer_accel_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(accel,mp->d_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40026);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_accel_from_device,
              TRNASFER_B_ACCEL_FROM_DEVICE)(int* size, float* b_accel,long* Mesh_pointer_f) {

TRACE("transfer_b_accel_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(b_accel,mp->d_b_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40036);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_sigma_from_device,
              TRANSFER_SIGMA_FROM_DEVICE)(int* size, float* sigma_kl,long* Mesh_pointer_f) {

TRACE("transfer_sigma_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(sigma_kl,mp->d_Sigma_kl,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40046);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_displ_from_device,
              TRANSFER_B_DISPL_FROM_DEVICE)(int* size, float* displ,long* Mesh_pointer_f) {

TRACE("transfer_b_displ_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40056);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_displ_from_device,
              TRANSFER_DISPL_FROM_DEVICE)(int* size, float* displ,long* Mesh_pointer_f) {

TRACE("transfer_displ_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),40066);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_compute_kernel_answers_from_device,
              TRANSFER_COMPUTE_KERNEL_ANSWERS_FROM_DEVICE)(long* Mesh_pointer,
                                                           float* rho_kl,int* size_rho,
                                                           float* mu_kl, int* size_mu,
                                                           float* kappa_kl, int* size_kappa) {
TRACE("transfer_compute_kernel_answers_from_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  cudaMemcpy(rho_kl,mp->d_rho_kl,*size_rho*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(mu_kl,mp->d_mu_kl,*size_mu*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(kappa_kl,mp->d_kappa_kl,*size_kappa*sizeof(float),cudaMemcpyDeviceToHost);  
  
}

/* ----------------------------------------------------------------------------------------------- */
/*
extern "C"
void FC_FUNC_(transfer_compute_kernel_fields_from_device,
              TRANSFER_COMPUTE_KERNEL_FIELDS_FROM_DEVICE)(long* Mesh_pointer,
                                                          float* accel, int* size_accel,
                                                          float* b_displ, int* size_b_displ,
                                                          float* epsilondev_xx,
                                                          float* epsilondev_yy,
                                                          float* epsilondev_xy,
                                                          float* epsilondev_xz,
                                                          float* epsilondev_yz,
                                                          int* size_epsilondev,
                                                          float* b_epsilondev_xx,
                                                          float* b_epsilondev_yy,
                                                          float* b_epsilondev_xy,
                                                          float* b_epsilondev_xz,
                                                          float* b_epsilondev_yz,
                                                          int* size_b_epsilondev,
                                                          float* rho_kl,int* size_rho,
                                                          float* mu_kl, int* size_mu,
                                                          float* kappa_kl, int* size_kappa,
                                                          float* epsilon_trace_over_3,
                                                          float* b_epsilon_trace_over_3,
                                                          int* size_epsilon_trace_over_3) {
TRACE("transfer_compute_kernel_fields_from_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  cudaMemcpy(accel,mp->d_accel,*size_accel*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_displ,mp->d_b_displ,*size_b_displ*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_xx,mp->d_epsilondev_xx,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_yy,mp->d_epsilondev_yy,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_xy,mp->d_epsilondev_xy,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_xz,mp->d_epsilondev_xz,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_yz,mp->d_epsilondev_yz,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_epsilondev_xx,mp->d_b_epsilondev_xx,*size_b_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_epsilondev_yy,mp->d_b_epsilondev_yy,*size_b_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_epsilondev_xy,mp->d_b_epsilondev_xy,*size_b_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_epsilondev_xz,mp->d_b_epsilondev_xz,*size_b_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_epsilondev_yz,mp->d_b_epsilondev_yz,*size_b_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(rho_kl,mp->d_rho_kl,*size_rho*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(mu_kl,mp->d_mu_kl,*size_mu*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(kappa_kl,mp->d_kappa_kl,*size_kappa*sizeof(float),cudaMemcpyDeviceToHost);  
  cudaMemcpy(epsilon_trace_over_3,mp->d_epsilon_trace_over_3,*size_epsilon_trace_over_3*sizeof(float),
	     cudaMemcpyDeviceToHost);
  cudaMemcpy(b_epsilon_trace_over_3,mp->d_b_epsilon_trace_over_3,*size_epsilon_trace_over_3*sizeof(float),
	     cudaMemcpyDeviceToHost);
       
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_compute_kernel_fields_from_device");
#endif
}
*/

/* ----------------------------------------------------------------------------------------------- */

// attenuation fields

extern "C"
void FC_FUNC_(transfer_b_fields_att_to_device,
              TRANSFER_B_FIELDS_ATT_TO_DEVICE)(long* Mesh_pointer,
                                             float* b_R_xx,float* b_R_yy,float* b_R_xy,float* b_R_xz,float* b_R_yz,
                                             int* size_R,
                                             float* b_epsilondev_xx,
                                             float* b_epsilondev_yy,
                                             float* b_epsilondev_xy,
                                             float* b_epsilondev_xz,
                                             float* b_epsilondev_yz,
                                             int* size_epsilondev) {
  TRACE("transfer_b_fields_att_to_device");
  //get mesh pointer out of fortran integer container
  Mesh* mp = (Mesh*)(*Mesh_pointer); 
  
  cudaMemcpy(mp->d_b_R_xx,b_R_xx,*size_R*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_R_yy,b_R_yy,*size_R*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_R_xy,b_R_xy,*size_R*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_R_xz,b_R_xz,*size_R*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_R_yz,b_R_yz,*size_R*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMemcpy(mp->d_b_epsilondev_xx,b_epsilondev_xx,*size_epsilondev*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_yy,b_epsilondev_yy,*size_epsilondev*sizeof(float),cudaMemcpyHostToDevice);  
  cudaMemcpy(mp->d_b_epsilondev_xy,b_epsilondev_xy,*size_epsilondev*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_xz,b_epsilondev_xz,*size_epsilondev*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_yz,b_epsilondev_yz,*size_epsilondev*sizeof(float),cudaMemcpyHostToDevice);
  
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_b_fields_att_to_device");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

// attenuation fields

extern "C"
void FC_FUNC_(transfer_fields_att_from_device,
              TRANSFER_FIELDS_ATT_FROM_DEVICE)(long* Mesh_pointer,
                                               float* R_xx,float* R_yy,float* R_xy,float* R_xz,float* R_yz,
                                               int* size_R,
                                               float* epsilondev_xx,
                                               float* epsilondev_yy,
                                               float* epsilondev_xy,
                                               float* epsilondev_xz,
                                               float* epsilondev_yz,
                                               int* size_epsilondev) {
  TRACE("transfer_fields_att_from_device");
  //get mesh pointer out of fortran integer container
  Mesh* mp = (Mesh*)(*Mesh_pointer); 

  cudaMemcpy(R_xx,mp->d_R_xx,*size_R*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(R_yy,mp->d_R_yy,*size_R*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(R_xy,mp->d_R_xy,*size_R*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(R_xz,mp->d_R_xz,*size_R*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(R_yz,mp->d_R_yz,*size_R*sizeof(float),cudaMemcpyDeviceToHost);
  
  cudaMemcpy(epsilondev_xx,mp->d_epsilondev_xx,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_yy,mp->d_epsilondev_yy,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_xy,mp->d_epsilondev_xy,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_xz,mp->d_epsilondev_xz,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(epsilondev_yz,mp->d_epsilondev_yz,*size_epsilondev*sizeof(float),cudaMemcpyDeviceToHost);
  
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_fields_att_from_device");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(transfer_sensitivity_kernels_to_host,
              TRANSFER_SENSITIVITY_KERNELS_TO_HOST)(long* Mesh_pointer, 
                                                    float* h_rho_kl,
                                                    float* h_mu_kl, 
                                                    float* h_kappa_kl,
                                                    int* NSPEC_AB) {
TRACE("transfer_sensitivity_kernels_to_host");
  //get mesh pointer out of fortran integer container
  Mesh* mp = (Mesh*)(*Mesh_pointer); 
  
  print_CUDA_error_if_any(cudaMemcpy(h_rho_kl,mp->d_rho_kl,*NSPEC_AB*125*sizeof(float),
                                     cudaMemcpyDeviceToHost),40101);
  print_CUDA_error_if_any(cudaMemcpy(h_mu_kl,mp->d_mu_kl,*NSPEC_AB*125*sizeof(float),
                                     cudaMemcpyDeviceToHost),40102);
  print_CUDA_error_if_any(cudaMemcpy(h_kappa_kl,mp->d_kappa_kl,*NSPEC_AB*125*sizeof(float),
                                     cudaMemcpyDeviceToHost),40103);
  
}

/* ----------------------------------------------------------------------------------------------- */

// for NOISE simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(transfer_sensitivity_kernels_noise_to_host,
              TRANSFER_SENSITIVITY_KERNELS_NOISE_TO_HOST)(long* Mesh_pointer, 
                                                          float* h_Sigma_kl,
                                                          int* NSPEC_AB) {
TRACE("transfer_sensitivity_kernels_noise_to_host");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(h_Sigma_kl,mp->d_Sigma_kl,125*(*NSPEC_AB)*sizeof(float),
                                     cudaMemcpyHostToDevice),40201);
  
}


/* ----------------------------------------------------------------------------------------------- */

// for ACOUSTIC simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_fields_acoustic_to_device,
              TRANSFER_FIELDS_ACOUSTIC_TO_DEVICE)(
                                                  int* size, 
                                                  float* potential_acoustic, 
                                                  float* potential_dot_acoustic, 
                                                  float* potential_dot_dot_acoustic,
                                                  long* Mesh_pointer_f) {
TRACE("transfer_fields_acoustic_to_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_potential_acoustic,potential_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),50110);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_potential_dot_acoustic,potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),50120);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_potential_dot_dot_acoustic,potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),50130);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_fields_acoustic_to_device");
#endif  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_fields_acoustic_to_device,
              TRANSFER_B_FIELDS_ACOUSTIC_TO_DEVICE)(
                                                    int* size, 
                                                    float* b_potential_acoustic, 
                                                    float* b_potential_dot_acoustic, 
                                                    float* b_potential_dot_dot_acoustic,
                                                    long* Mesh_pointer_f) {
TRACE("transfer_b_fields_acoustic_to_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_b_potential_acoustic,b_potential_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),51110);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_b_potential_dot_acoustic,b_potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),51120);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_b_potential_dot_dot_acoustic,b_potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),51130);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_b_fields_acoustic_to_device");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_fields_acoustic_from_device,TRANSFER_FIELDS_ACOUSTIC_FROM_DEVICE)(
                                                                                         int* size, 
                                                                                         float* potential_acoustic, 
                                                                                         float* potential_dot_acoustic, 
                                                                                         float* potential_dot_dot_acoustic,
                                                                                         long* Mesh_pointer_f) {
TRACE("transfer_fields_acoustic_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(potential_acoustic,mp->d_potential_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),52111);
  print_CUDA_error_if_any(cudaMemcpy(potential_dot_acoustic,mp->d_potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),52121);
  print_CUDA_error_if_any(cudaMemcpy(potential_dot_dot_acoustic,mp->d_potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),52131);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_fields_acoustic_from_device");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_fields_acoustic_from_device,
              TRANSFER_B_FIELDS_ACOUSTIC_FROM_DEVICE)(
                                                      int* size, 
                                                      float* b_potential_acoustic, 
                                                      float* b_potential_dot_acoustic, 
                                                      float* b_potential_dot_dot_acoustic,
                                                      long* Mesh_pointer_f) {
TRACE("transfer_b_fields_acoustic_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(b_potential_acoustic,mp->d_b_potential_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),53111);
  print_CUDA_error_if_any(cudaMemcpy(b_potential_dot_acoustic,mp->d_b_potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),53121);
  print_CUDA_error_if_any(cudaMemcpy(b_potential_dot_dot_acoustic,mp->d_b_potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),53131);  
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_b_fields_acoustic_from_device");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(transfer_sensitivity_kernels_acoustic_to_host,
              TRANSFER_SENSITIVITY_KERNELS_ACOUSTIC_TO_HOST)(long* Mesh_pointer, 
                                                             float* h_rho_ac_kl,
                                                             float* h_kappa_ac_kl,
                                                             int* NSPEC_AB) {
  
  TRACE("transfer_sensitivity_kernels_acoustic_to_host");  
  
  //get mesh pointer out of fortran integer container  
  Mesh* mp = (Mesh*)(*Mesh_pointer); 
  int size = *NSPEC_AB*125;
  
  // copies kernel values over to CPU host
  print_CUDA_error_if_any(cudaMemcpy(h_rho_ac_kl,mp->d_rho_ac_kl,size*sizeof(float),
                                     cudaMemcpyDeviceToHost),54101);
  print_CUDA_error_if_any(cudaMemcpy(h_kappa_ac_kl,mp->d_kappa_ac_kl,size*sizeof(float),
                                     cudaMemcpyDeviceToHost),54102);  
}

