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

/* daniel: trivia

- for most real working arrays we use float (single exception so far is stf_pre_compute).
  TODO: we could use "realw" instead of "float" type declaration to make it easier to switch
              between a real or double precision simulation (matching CUSTOM_REAL == 4 or 8 in fortran routines).

- instead of boolean "logical" declared in fortran routines, in C (or Cuda-C) we have to use "int" variables.

  ifort / gfortran caveat:
    to check whether it is true or false, do not check for == 1 to test for true values since ifort just uses
    non-zero values for true (e.g. can be -1 for true). however, false will be always == 0.

  thus, rather use: if( var ) {...}  for testing if true instead of if( var == 1){...} (alternative: one could use if( var != 0 ){...}

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
#define TRACE(x) printf("%s\n",x);
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

// error checking after cuda function calls
#define ENABLE_VERY_SLOW_ERROR_CHECKING

#define MAX(x,y)                    (((x) < (y)) ? (y) : (x))

double get_time();

void print_CUDA_error_if_any(cudaError_t err, int num);

void pause_for_debugger(int pause);

void exit_on_cuda_error(char* kernel_name);

void exit_on_error(char* info);

/* ----------------------------------------------------------------------------------------------- */

// cuda constant arrays

/* ----------------------------------------------------------------------------------------------- */

// dimensions
#define NDIM 3

// Gauss-Lobatto-Legendre 
#define NGLLX 5
#define NGLL2 25
#define NGLL3 125 // no padding: requires same size as in fortran for NGLLX * NGLLY * NGLLZ

// padding: 128 == 2**7 might improve on older graphics cards w/ coalescent memory accesses:
#define NGLL3_PADDED 128
// no padding: 125 == 5*5*5 to avoid allocation of extra memory
//#define NGLL3_PADDED 125

// number of standard linear solids
#define N_SLS 3

//typedef float real;   // type of variables passed into function
typedef float realw;  // type of "working" variables

// double precision temporary variables leads to 10% performance
// decrease in Kernel_2_impl (not very much..)
typedef float reald;

// (optional) pre-processing directive used in kernels: if defined check that it is also set in src/shared/constants.h:
// leads up to ~ 5% performance increase
//#define USE_MESH_COLORING_GPU

// cuda kernel block size for updating displacements/potential (newmark time scheme)
#define BLOCKSIZE_KERNEL1 128
#define BLOCKSIZE_KERNEL3 128
#define BLOCKSIZE_TRANSFER 256

/* ----------------------------------------------------------------------------------------------- */

// indexing

#define INDEX2(xsize,x,y) x + (y)*xsize

#define INDEX3(xsize,ysize,x,y,z) x + xsize*(y + ysize*z)
//#define INDEX3(xsize,ysize,x,y,z) x + (y)*xsize + (z)*xsize*ysize

#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*(z + zsize*i))
//#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize

#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + xsize*(y + ysize*(z + zsize*(i + isize*(j))))
//#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize + (j)*xsize*ysize*zsize*isize

#define INDEX6(xsize,ysize,zsize,isize,jsize,x,y,z,i,j,k) x + xsize*(y + ysize*(z + zsize*(i + isize*(j + jsize*k))))

#define INDEX4_PADDED(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*z) + (i)*NGLL3_PADDED
//#define INDEX4_PADDED(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*NGLL3_PADDED

/* ----------------------------------------------------------------------------------------------- */

// mesh pointer wrapper structure

/* ----------------------------------------------------------------------------------------------- */

typedef struct mesh_ {

  // mesh resolution
  int NSPEC_AB;
  int NGLOB_AB;

  // interpolators
  realw* d_xix; realw* d_xiy; realw* d_xiz;
  realw* d_etax; realw* d_etay; realw* d_etaz;
  realw* d_gammax; realw* d_gammay; realw* d_gammaz;

  // model parameters
  realw* d_kappav; realw* d_muv;

  // global indexing
  int* d_ibool;

  // inner / outer elements
  int* d_ispec_is_inner;

  // mesh coloring
  int use_mesh_coloring_gpu;

  // pointers to constant memory arrays
  realw* d_hprime_xx; realw* d_hprime_yy; realw* d_hprime_zz;
  realw* d_hprimewgll_xx; realw* d_hprimewgll_yy; realw* d_hprimewgll_zz;
  realw* d_wgllwgll_xy; realw* d_wgllwgll_xz; realw* d_wgllwgll_yz;

  // mpi buffers
  int num_interfaces_ext_mesh;
  int max_nibool_interfaces_ext_mesh;
  
  // ------------------------------------------------------------------ //
  // elastic wavefield parameters
  // ------------------------------------------------------------------ //

  // displacement, velocity, acceleration
  realw* d_displ; realw* d_veloc; realw* d_accel;
  // backward/reconstructed elastic wavefield
  realw* d_b_displ; realw* d_b_veloc; realw* d_b_accel;

  // elastic elements
  int* d_ispec_is_elastic;

  // elastic domain parameters
  int* d_phase_ispec_inner_elastic;
  int num_phase_ispec_elastic;

  // mesh coloring
  int* h_num_elem_colors_elastic;
  int num_colors_outer_elastic,num_colors_inner_elastic;
  int nspec_elastic;

  realw* d_rmass;
  
  // mpi buffer
  realw* d_send_accel_buffer;

  // interfaces
  int* d_nibool_interfaces_ext_mesh;
  int* d_ibool_interfaces_ext_mesh;

  //used for absorbing stacey boundaries
  int d_num_abs_boundary_faces;
  int* d_abs_boundary_ispec;
  int* d_abs_boundary_ijk;
  realw* d_abs_boundary_normal;
  realw* d_abs_boundary_jacobian2Dw;

  realw* d_b_absorb_field;
  int d_b_reclen_field;

  realw* d_rho_vp;
  realw* d_rho_vs;

  // sources
  int nsources_local;
  realw* d_sourcearrays;
  double* d_stf_pre_compute;
  int* d_islice_selected_source;
  int* d_ispec_selected_source;

  // receivers
  int* d_number_receiver_global;
  int* d_ispec_selected_rec;
  int* d_islice_selected_rec;
  int nrec_local;
  realw* d_station_seismo_field;
  realw* h_station_seismo_field;

  // adjoint receivers/sources
  int nadj_rec_local;
  realw* d_adj_sourcearrays;
  realw* h_adj_sourcearrays_slice;
  int* d_pre_computed_irec;

  // surface elements (to save for noise tomography and acoustic simulations)
  int* d_free_surface_ispec;
  int* d_free_surface_ijk;
  int num_free_surface_faces;

  // surface movie elements to save for noise tomography
  realw* d_noise_surface_movie;

  // attenuation
  realw* d_R_xx;
  realw* d_R_yy;
  realw* d_R_xy;
  realw* d_R_xz;
  realw* d_R_yz;

  realw* d_one_minus_sum_beta;
  realw* d_factor_common;

  realw* d_alphaval;
  realw* d_betaval;
  realw* d_gammaval;

  // attenuation & kernel
  realw* d_epsilondev_xx;
  realw* d_epsilondev_yy;
  realw* d_epsilondev_xy;
  realw* d_epsilondev_xz;
  realw* d_epsilondev_yz;
  realw* d_epsilon_trace_over_3;

  // anisotropy
  realw* d_c11store;
  realw* d_c12store;
  realw* d_c13store;
  realw* d_c14store;
  realw* d_c15store;
  realw* d_c16store;
  realw* d_c22store;
  realw* d_c23store;
  realw* d_c24store;
  realw* d_c25store;
  realw* d_c26store;
  realw* d_c33store;
  realw* d_c34store;
  realw* d_c35store;
  realw* d_c36store;
  realw* d_c44store;
  realw* d_c45store;
  realw* d_c46store;
  realw* d_c55store;
  realw* d_c56store;
  realw* d_c66store;

  // noise
  realw* d_normal_x_noise;
  realw* d_normal_y_noise;
  realw* d_normal_z_noise;
  realw* d_mask_noise;
  realw* d_free_surface_jacobian2Dw;

  realw* d_noise_sourcearray;

  // attenuation & kernel backward fields
  realw* d_b_R_xx;
  realw* d_b_R_yy;
  realw* d_b_R_xy;
  realw* d_b_R_xz;
  realw* d_b_R_yz;

  realw* d_b_epsilondev_xx;
  realw* d_b_epsilondev_yy;
  realw* d_b_epsilondev_xy;
  realw* d_b_epsilondev_xz;
  realw* d_b_epsilondev_yz;
  realw* d_b_epsilon_trace_over_3;

  realw* d_b_alphaval;
  realw* d_b_betaval;
  realw* d_b_gammaval;

  // sensitivity kernels
  realw* d_rho_kl;
  realw* d_mu_kl;
  realw* d_kappa_kl;

  // noise sensitivity kernel
  realw* d_Sigma_kl;

  // approximative hessian for preconditioning kernels
  realw* d_hess_el_kl;

  // oceans
  realw* d_rmass_ocean_load;
  realw* d_free_surface_normal;
  int* d_updated_dof_ocean_load;

  // ------------------------------------------------------------------ //
  // acoustic wavefield
  // ------------------------------------------------------------------ //
  // potential and first and second time derivative
  realw* d_potential_acoustic; realw* d_potential_dot_acoustic; realw* d_potential_dot_dot_acoustic;
  // backward/reconstructed wavefield
  realw* d_b_potential_acoustic; realw* d_b_potential_dot_acoustic; realw* d_b_potential_dot_dot_acoustic;

  // acoustic domain parameters
  int* d_ispec_is_acoustic;

  int* d_phase_ispec_inner_acoustic;
  int num_phase_ispec_acoustic;

  // mesh coloring
  int* h_num_elem_colors_acoustic;
  int num_colors_outer_acoustic,num_colors_inner_acoustic;
  int nspec_acoustic;

  realw* d_rhostore;
  realw* d_kappastore;
  realw* d_rmass_acoustic;
  
  // mpi buffer
  realw* d_send_potential_dot_dot_buffer;

  realw* d_b_absorb_potential;
  int d_b_reclen_potential;

  // for writing seismograms
  realw* d_station_seismo_potential;
  realw* h_station_seismo_potential;

  // sensitivity kernels
  realw* d_rho_ac_kl;
  realw* d_kappa_ac_kl;

  // approximative hessian for preconditioning kernels
  realw* d_hess_ac_kl;

  // coupling acoustic-elastic
  int* d_coupling_ac_el_ispec;
  int* d_coupling_ac_el_ijk;
  realw* d_coupling_ac_el_normal;
  realw* d_coupling_ac_el_jacobian2Dw;

} Mesh;


#endif
