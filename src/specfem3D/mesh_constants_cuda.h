#ifndef GPU_MESH_
#define GPU_MESH_
#include <sys/types.h>
#include <unistd.h>

typedef struct mesh_ {
  
  int NGLLX; int NSPEC_AB;
  int NGLOB_AB;
  float* d_xix; float* d_xiy; float* d_xiz;
  float* d_etax; float* d_etay; float* d_etaz;
  float* d_gammax; float* d_gammay; float* d_gammaz;
  float* d_kappav; float* d_muv;
  int* d_ibool;
  float* d_displ; float* d_veloc; float* d_accel;
  float* d_b_displ; float* d_b_veloc; float* d_b_accel;
  int* d_phase_ispec_inner_elastic;
  int d_num_phase_ispec_elastic;
  float* d_rmass;
  float* d_send_accel_buffer;
  int* d_nibool_interfaces_ext_mesh;
  int* d_ibool_interfaces_ext_mesh;

  // used for writing seismograms
  int* d_number_receiver_global;
  int* d_ispec_selected_rec;
  int nrec_local;
  float* d_station_seismo_field;
  float* h_station_seismo_field;
    
  //used for absorbing stacey boundaries
  int* d_abs_boundary_ispec;
  int* d_abs_boundary_ijk;
  float* d_abs_boundary_normal;
  float* d_rho_vp;
  float* d_rho_vs;
  float* d_abs_boundary_jacobian2Dw;
  float* d_b_absorb_field;
  int* d_ispec_is_inner;
  int* d_ispec_is_elastic;
  float* d_sourcearrays;
  double* d_stf_pre_compute;
  int* d_islice_selected_source;
  int* d_ispec_selected_source;

  int* d_islice_selected_rec;
  
  // surface elements to save for noise tomography
  int* d_free_surface_ispec;
  int* d_free_surface_ijk;
  int num_free_surface_faces;
  float* d_noise_surface_movie;

  float* d_epsilondev_xx;
  float* d_epsilondev_yy;
  float* d_epsilondev_xy;
  float* d_epsilondev_xz;
  float* d_epsilondev_yz;
  float* d_epsilon_trace_over_3;
  
  float* d_normal_x_noise;
  float* d_normal_y_noise;
  float* d_normal_z_noise;
  float* d_mask_noise;
  float* d_free_surface_jacobian2Dw;

  float* d_wgllwgll_xy;
  float* d_wgllwgll_xz;
  float* d_wgllwgll_yz;

  float* d_noise_sourcearray;

  float* d_b_epsilondev_xx;
  float* d_b_epsilondev_yy;
  float* d_b_epsilondev_xy;
  float* d_b_epsilondev_xz;
  float* d_b_epsilondev_yz;
  float* d_b_epsilon_trace_over_3;
  
  // sensitivity kernels
  float* d_rho_kl;
  float* d_mu_kl;
  float* d_kappa_kl;
  float* d_Sigma_kl;

  
} Mesh;

void pause_for_debugger(int pause); 

void exit_on_cuda_error(char* kernel_name);

#endif
