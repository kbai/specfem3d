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

  print_CUDA_error_if_any(cudaMemcpy(mp->d_displ,displ,sizeof(float)*(*size),cudaMemcpyHostToDevice),3);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_veloc,veloc,sizeof(float)*(*size),cudaMemcpyHostToDevice),4);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_accel,accel,sizeof(float)*(*size),cudaMemcpyHostToDevice),5);
  
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
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);
  print_CUDA_error_if_any(cudaMemcpy(veloc,mp->d_veloc,sizeof(float)*(*size),cudaMemcpyDeviceToHost),7);
  print_CUDA_error_if_any(cudaMemcpy(accel,mp->d_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),8);
  
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
  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_accel,accel,sizeof(float)*(*size),cudaMemcpyHostToDevice),6);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_accel_from_device,
              TRANSFER_ACCEL_FROM_DEVICE)(int* size, float* accel,long* Mesh_pointer_f) {

TRACE("transfer_accel_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(accel,mp->d_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_accel_from_device,
              TRNASFER_B_ACCEL_FROM_DEVICE)(int* size, float* b_accel,long* Mesh_pointer_f) {

TRACE("transfer_b_accel_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(b_accel,mp->d_b_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_sigma_from_device,
              TRANSFER_SIGMA_FROM_DEVICE)(int* size, float* sigma_kl,long* Mesh_pointer_f) {

TRACE("transfer_sigma_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(sigma_kl,mp->d_Sigma_kl,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_b_displ_from_device,
              TRANSFER_B_DISPL_FROM_DEVICE)(int* size, float* displ,long* Mesh_pointer_f) {

TRACE("transfer_b_displ_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_displ_from_device,
              TRANSFER_DISPL_FROM_DEVICE)(int* size, float* displ,long* Mesh_pointer_f) {

TRACE("transfer_displ_from_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

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


/* ----------------------------------------------------------------------------------------------- */

// for ACOUSTIC simulations

/* ----------------------------------------------------------------------------------------------- */

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
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),110);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_potential_dot_acoustic,potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),120);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_potential_dot_dot_acoustic,potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),130);
  
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
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),110);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_b_potential_dot_acoustic,b_potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),120);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_b_potential_dot_dot_acoustic,b_potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyHostToDevice),130);
  
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
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),111);
  print_CUDA_error_if_any(cudaMemcpy(potential_dot_acoustic,mp->d_potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),121);
  print_CUDA_error_if_any(cudaMemcpy(potential_dot_dot_acoustic,mp->d_potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),131);
  
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
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),111);
  print_CUDA_error_if_any(cudaMemcpy(b_potential_dot_acoustic,mp->d_b_potential_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),121);
  print_CUDA_error_if_any(cudaMemcpy(b_potential_dot_dot_acoustic,mp->d_b_potential_dot_dot_acoustic,
                                     sizeof(float)*(*size),cudaMemcpyDeviceToHost),131);  
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after transfer_b_fields_acoustic_from_device");
#endif
}


