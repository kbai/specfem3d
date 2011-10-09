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

#ifndef GPU_MESH_
#define GPU_MESH_
#include <sys/types.h>
#include <unistd.h>


/* ----------------------------------------------------------------------------------------------- */

// for debugging and benchmarking

/* ----------------------------------------------------------------------------------------------- */

#define DEBUG 0
#if DEBUG == 1
#define TRACE(x) printf("%s\n",x)
#else
#define TRACE(x) // printf("%s\n",x);
#endif

#define MAXDEBUG 0
#if MAXDEBUG == 1
#define LOG(x) printf("%s\n",x)
#define PRINT5(var,offset) for(;print_count<5;print_count++) printf("var(%d)=%2.20f\n",print_count,var[offset+print_count]);
#define PRINT10(var) if(print_count<10) { printf("var=%1.20e\n",var); print_count++; }
#define PRINT10i(var) if(print_count<10) { printf("var=%d\n",var); print_count++; }
#else
#define LOG(x) // printf("%s\n",x);
#define PRINT5(var,offset) // for(i=0;i<10;i++) printf("var(%d)=%f\n",i,var[offset+i]);
#endif


#define ENABLE_VERY_SLOW_ERROR_CHECKING

/* ----------------------------------------------------------------------------------------------- */

// indexing

#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + (y)*xsize + (z)*xsize*ysize
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize
#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize + (j)*xsize*ysize*zsize*isize
#define INDEX6(xsize,ysize,zsize,isize,jsize,x,y,z,i,j,k) x + xsize*(y + ysize*(z + zsize*(i + isize*(j + jsize*k))))

#define INDEX4_PADDED(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*128

//daniel: check speed of alternatives
//#define INDEX2(xsize,x,y) x + (y)*xsize
//#define INDEX3(xsize,ysize,x,y,z) x + xsize*(y + ysize*z)
//#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*(z + zsize*i))
//#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + xsize*(y + ysize*(z + zsize*(i + isize*j)))

/* ----------------------------------------------------------------------------------------------- */

#define MAX(x,y)                    (((x) < (y)) ? (y) : (x))

double get_time();

void print_CUDA_error_if_any(cudaError_t err, int num);

void pause_for_debugger(int pause); 

void exit_on_cuda_error(char* kernel_name);

/* ----------------------------------------------------------------------------------------------- */

// cuda constant arrays

/* ----------------------------------------------------------------------------------------------- */

#define NDIM 3
#define NGLLX 5
#define NGLL2 25
#define N_SLS 3

typedef float real;   // type of variables passed into function
typedef float realw;  // type of "working" variables

// double precision temporary variables leads to 10% performance
// decrease in Kernel_2_impl (not very much..)
typedef float reald;

/* ----------------------------------------------------------------------------------------------- */

// mesh pointer wrapper structure

/* ----------------------------------------------------------------------------------------------- */

typedef struct mesh_ {

  // mesh resolution
  int NSPEC_AB;
  int NGLOB_AB;
  
  // interpolators
  float* d_xix; float* d_xiy; float* d_xiz;
  float* d_etax; float* d_etay; float* d_etaz;
  float* d_gammax; float* d_gammay; float* d_gammaz;

  // model parameters  
  float* d_kappav; float* d_muv;

  // global indexing  
  int* d_ibool;

  // pointers to constant memory arrays
  float* d_hprime_xx; float* d_hprime_yy; float* d_hprime_zz;
  float* d_hprimewgll_xx; float* d_hprimewgll_yy; float* d_hprimewgll_zz;
  float* d_wgllwgll_xy; float* d_wgllwgll_xz; float* d_wgllwgll_yz;

  // ------------------------------------------------------------------ //
  // elastic wavefield parameters
  // ------------------------------------------------------------------ //
  
  // displacement, velocity, acceleration  
  float* d_displ; float* d_veloc; float* d_accel;
  // backward/reconstructed elastic wavefield  
  float* d_b_displ; float* d_b_veloc; float* d_b_accel;

  // elastic domain parameters    
  int* d_phase_ispec_inner_elastic;
  int d_num_phase_ispec_elastic;
  float* d_rmass;
  float* d_send_accel_buffer;

  // interfaces  
  int* d_nibool_interfaces_ext_mesh;
  int* d_ibool_interfaces_ext_mesh;
    
  //used for absorbing stacey boundaries
  int d_num_abs_boundary_faces;
  int* d_abs_boundary_ispec;
  int* d_abs_boundary_ijk;
  float* d_abs_boundary_normal;
  float* d_abs_boundary_jacobian2Dw;

  float* d_b_absorb_field;
  int d_b_reclen_field;
  
  float* d_rho_vp;
  float* d_rho_vs;
  
  // inner / outer elements  
  int* d_ispec_is_inner;
  int* d_ispec_is_elastic;

  // sources  
  float* d_sourcearrays;
  double* d_stf_pre_compute;
  int* d_islice_selected_source;
  int* d_ispec_selected_source;

  // receivers
  int* d_number_receiver_global;
  int* d_ispec_selected_rec;  
  int* d_islice_selected_rec;
  int nrec_local;
  float* d_station_seismo_field;
  float* h_station_seismo_field;  
  
  // surface elements (to save for noise tomography and acoustic simulations)
  int* d_free_surface_ispec;
  int* d_free_surface_ijk;
  int num_free_surface_faces;
  
  // surface movie elements to save for noise tomography
  float* d_noise_surface_movie;

  // attenuation
  float* d_R_xx;
  float* d_R_yy;
  float* d_R_xy;
  float* d_R_xz;
  float* d_R_yz;

  float* d_one_minus_sum_beta;
  float* d_factor_common;
  
  float* d_alphaval;
  float* d_betaval;
  float* d_gammaval;
  
  // attenuation & kernel
  float* d_epsilondev_xx;
  float* d_epsilondev_yy;
  float* d_epsilondev_xy;
  float* d_epsilondev_xz;
  float* d_epsilondev_yz;
  float* d_epsilon_trace_over_3;
      
  // noise
  float* d_normal_x_noise;
  float* d_normal_y_noise;
  float* d_normal_z_noise;
  float* d_mask_noise;
  float* d_free_surface_jacobian2Dw;

  float* d_noise_sourcearray;

  // attenuation & kernel backward fields
  float* d_b_R_xx;
  float* d_b_R_yy;
  float* d_b_R_xy;
  float* d_b_R_xz;
  float* d_b_R_yz;
  
  float* d_b_epsilondev_xx;
  float* d_b_epsilondev_yy;
  float* d_b_epsilondev_xy;
  float* d_b_epsilondev_xz;
  float* d_b_epsilondev_yz;
  float* d_b_epsilon_trace_over_3;

  float* d_b_alphaval;
  float* d_b_betaval;
  float* d_b_gammaval;
  
  // sensitivity kernels
  float* d_rho_kl;
  float* d_mu_kl;
  float* d_kappa_kl;
  
  // noise sensitivity kernel
  float* d_Sigma_kl;

  // oceans
  float* d_rmass_ocean_load;
  float* d_free_surface_normal;
  
  // ------------------------------------------------------------------ //
  // acoustic wavefield
  // ------------------------------------------------------------------ //
  // potential and first and second time derivative
  float* d_potential_acoustic; float* d_potential_dot_acoustic; float* d_potential_dot_dot_acoustic;
  // backward/reconstructed wavefield
  float* d_b_potential_acoustic; float* d_b_potential_dot_acoustic; float* d_b_potential_dot_dot_acoustic;
  
  // acoustic domain parameters  
  int* d_phase_ispec_inner_acoustic;  
  int num_phase_ispec_acoustic;
  
  float* d_rhostore;
  float* d_kappastore;
  float* d_rmass_acoustic;
  
  float* d_send_potential_dot_dot_buffer;
  int* d_ispec_is_acoustic;
    
  float* d_b_absorb_potential;
  int d_b_reclen_potential;
  
  // for writing seismograms
  float* d_station_seismo_potential;
  float* h_station_seismo_potential;
  
  // sensitivity kernels
  float* d_rho_ac_kl;
  float* d_kappa_ac_kl;
  
  // coupling acoustic-elastic
  int* d_coupling_ac_el_ispec;
  int* d_coupling_ac_el_ijk;
  float* d_coupling_ac_el_normal;
  float* d_coupling_ac_el_jacobian2Dw;


  
} Mesh;


#endif
