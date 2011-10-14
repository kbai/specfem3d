#include "config.h"

typedef float real;

/* from check_fields_cuda.cu */
void FC_FUNC_(check_max_norm_displ_gpu,
              CHECK_MAX_NORM_DISPL_GPU)(int* size, float* displ,long* Mesh_pointer_f,int* announceID){}
				       
void FC_FUNC_(check_max_norm_vector,
              CHECK_MAX_NORM_VECTOR)(int* size, float* vector1, int* announceID){}				       
void FC_FUNC_(check_max_norm_displ,
              CHECK_MAX_NORM_DISPL)(int* size, float* displ, int* announceID){}

void FC_FUNC_(check_max_norm_b_displ_gpu,
              CHECK_MAX_NORM_B_DISPL_GPU)(int* size, float* b_displ,long* Mesh_pointer_f,int* announceID){}

void FC_FUNC_(check_max_norm_b_accel_gpu,
              CHECK_MAX_NORM_B_ACCEL_GPU)(int* size, float* b_accel,long* Mesh_pointer_f,int* announceID){}

void FC_FUNC_(check_max_norm_b_veloc_gpu,
              CHECK_MAX_NORM_B_VELOC_GPU)(int* size, float* b_veloc,long* Mesh_pointer_f,int* announceID){}

void FC_FUNC_(check_max_norm_b_displ,
              CHECK_MAX_NORM_B_DISPL)(int* size, float* b_displ,int* announceID){}

void FC_FUNC_(check_max_norm_b_accel,
              CHECK_MAX_NORM_B_ACCEL)(int* size, float* b_accel,int* announceID){}

void FC_FUNC_(check_error_vectors,
              CHECK_ERROR_VECTORS)(int* sizef, float* vector1,float* vector2){}

void FC_FUNC_(get_max_accel,
              GET_MAX_ACCEL)(int* itf,int* sizef,long* Mesh_pointer){}

void FC_FUNC_(get_norm_acoustic_from_device_cuda,
              GET_NORM_ACOUSTIC_FROM_DEVICE_CUDA)(float* norm, 
                                                  long* Mesh_pointer_f,
                                                  int* SIMULATION_TYPE){}

void FC_FUNC_(get_norm_elastic_from_device_cuda,
              GET_NORM_ELASTIC_FROM_DEVICE_CUDA)(float* norm, 
                                                 long* Mesh_pointer_f,
                                                 int* SIMULATION_TYPE){}

						
/* from file compute_add_sources_cuda.cu */

void FC_FUNC_(add_sourcearrays_adjoint_cuda,
              ADD_SOURCEARRAYS_ADJOINT_CUDA)(long* Mesh_pointer,
					     int* USE_FORCE_POINT_SOURCE,
					     double* h_stf_pre_compute,int* NSOURCES,
					     int* phase_is_inner,int* myrank){}

void FC_FUNC_(compute_add_sources_elastic_cuda,
              COMPUTE_ADD_SOURCES_ELASTIC_CUDA)(){}

void FC_FUNC_(add_source_master_rec_noise_cuda,
              ADD_SOURCE_MASTER_REC_NOISE_CUDA)(long* Mesh_pointer_f, 
                                                int* myrank_f,  
                                                int* it_f, 
                                                int* irec_master_noise_f, 
                                                int* islice_selected_rec){}

void FC_FUNC_(add_sources_sim_type_2_or_3,
              ADD_SOURCES_SIM_TYPE_2_OR_3)(long* Mesh_pointer, 
                                           float* h_adj_sourcearrays,
                                           int* size_adj_sourcearrays, int* ispec_is_inner,
                                           int* phase_is_inner, int* ispec_selected_rec,
                                           int* ibool,
                                           int* myrank, int* nrec, int* time_index,
                                           int* h_islice_selected_rec,int* nadj_rec_local,
                                           int* NTSTEP_BETWEEN_READ_ADJSRC){}

void FC_FUNC_(compute_add_sources_acoustic_cuda,
              COMPUTE_ADD_SOURCES_ACOUSTIC_CUDA)(long* Mesh_pointer_f, 
                                                 int* phase_is_innerf,
                                                 int* NSOURCESf, 
                                                 int* SIMULATION_TYPEf,
                                                 int* USE_FORCE_POINT_SOURCEf, 
                                                 double* h_stf_pre_compute, 
                                                 int* myrankf){}

void FC_FUNC_(compute_add_sources_acoustic_sim3_cuda,
              COMPUTE_ADD_SOURCES_ACOUSTIC_SIM3_CUDA)(long* Mesh_pointer_f, 
                                                      int* phase_is_innerf,
                                                      int* NSOURCESf, 
                                                      int* SIMULATION_TYPEf,
                                                      int* USE_FORCE_POINT_SOURCEf, 
                                                      double* h_stf_pre_compute, 
                                                      int* myrankf){}

void FC_FUNC_(add_sources_acoustic_sim_type_2_or_3_cuda,
              ADD_SOURCES_ACOUSTIC_SIM_TYPE_2_OR_3_CUDA)(long* Mesh_pointer, 
                                                         float* h_adj_sourcearrays,
                                                         int* size_adj_sourcearrays, 
                                                         int* phase_is_inner,
                                                         int* myrank, 
                                                         int* nrec, 
                                                         int* pre_computed_index,
                                                         int* h_islice_selected_rec,
                                                         int* nadj_rec_local,
                                                         int* NTSTEP_BETWEEN_ADJSRC){}

/* from compute_coupling_cuda.cu */
							
void FC_FUNC_(compute_coupling_acoustic_el_cuda,
              COMPUTE_COUPLING_ACOUSTIC_EL_CUDA)(
                                            long* Mesh_pointer_f, 
                                            int* phase_is_innerf, 
                                            int* num_coupling_ac_el_facesf, 
                                            int* SIMULATION_TYPEf){}

void FC_FUNC_(compute_coupling_elastic_ac_cuda,
              COMPUTE_COUPLING_ELASTIC_AC_CUDA)(
                                                 long* Mesh_pointer_f, 
                                                 int* phase_is_innerf, 
                                                 int* num_coupling_ac_el_facesf, 
                                                 int* SIMULATION_TYPEf){}

/* from compute_forces_acoustic_cuda.cu */

void FC_FUNC_(transfer_boundary_potential_from_device,
              TRANSFER_BOUNDARY_POTENTIAL_FROM_DEVICE)(
                                              int* size, 
                                              long* Mesh_pointer_f, 
                                              float* potential_dot_dot_acoustic, 
                                              float* send_potential_dot_dot_buffer,
                                              int* num_interfaces_ext_mesh, 
                                              int* max_nibool_interfaces_ext_mesh,
                                              int* nibool_interfaces_ext_mesh, 
                                              int* ibool_interfaces_ext_mesh,
                                              int* FORWARD_OR_ADJOINT){}

void FC_FUNC_(transfer_and_assemble_potential_to_device,
              TRANSFER_AND_ASSEMBLE_POTENTIAL_TO_DEVICE)(
                                                long* Mesh_pointer, 
                                                real* potential_dot_dot_acoustic, 
                                                real* buffer_recv_scalar_ext_mesh,
                                                int* num_interfaces_ext_mesh, 
                                                int* max_nibool_interfaces_ext_mesh,
                                                int* nibool_interfaces_ext_mesh, 
                                                int* ibool_interfaces_ext_mesh,
                                                int* FORWARD_OR_ADJOINT){}

void FC_FUNC_(compute_forces_acoustic_cuda,
              COMPUTE_FORCES_ACOUSTIC_CUDA)(long* Mesh_pointer_f,
                                            int* iphase,
                                            int* nspec_outer_acoustic,
                                            int* nspec_inner_acoustic,
                                            int* SIMULATION_TYPE){}

void FC_FUNC_(kernel_3_a_acoustic_cuda,KERNEL_3_ACOUSTIC_CUDA)(
                             long* Mesh_pointer,
                             int* size_F, 
                             int* SIMULATION_TYPE){}

void FC_FUNC_(kernel_3_b_acoustic_cuda,KERNEL_3_ACOUSTIC_CUDA)(
                                                             long* Mesh_pointer,
                                                             int* size_F, 
                                                             float* deltatover2_F, 
                                                             int* SIMULATION_TYPE, 
                                                             float* b_deltatover2_F){}

void FC_FUNC_(acoustic_enforce_free_surface_cuda,
              ACOUSTIC_ENFORCE_FREE_SURFACE_CUDA)(long* Mesh_pointer_f, 
                                                  int* SIMULATION_TYPE,
                                                  int* ABSORB_FREE_SURFACE){}


/* from compute_forces_elastic_cuda.cu */
void FC_FUNC_(transfer_boundary_accel_from_device,
              TRANSFER_BOUNDARY_ACCEL_FROM_DEVICE)(int* size, long* Mesh_pointer_f, float* accel,
						   float* send_accel_buffer,
						   int* num_interfaces_ext_mesh,
						   int* max_nibool_interfaces_ext_mesh,
						   int* nibool_interfaces_ext_mesh,
						   int* ibool_interfaces_ext_mesh,
						   int* FORWARD_OR_ADJOINT){}

						  
void FC_FUNC_(transfer_and_assemble_accel_to_device,
              TRANSFER_AND_ASSEMBLE_ACCEL_TO_DEVICE)(long* Mesh_pointer, real* accel,
                                                    real* buffer_recv_vector_ext_mesh,
                                                    int* num_interfaces_ext_mesh,
                                                    int* max_nibool_interfaces_ext_mesh,
                                                    int* nibool_interfaces_ext_mesh,
						     int* ibool_interfaces_ext_mesh,int* FORWARD_OR_ADJOINT){}

void FC_FUNC_(compute_forces_elastic_cuda,
              COMPUTE_FORCES_ELASTIC_CUDA)(long* Mesh_pointer_f,
                                           int* iphase,
                                           int* nspec_outer_elastic,
                                           int* nspec_inner_elastic,
                                           int* COMPUTE_AND_STORE_STRAIN,
                                           int* SIMULATION_TYPE,
                                           int* ATTENUATION){}

void FC_FUNC_(kernel_3_a_cuda,
              KERNEL_3_A_CUDA)(long* Mesh_pointer,
                               int* size_F, 
                               float* deltatover2_F, 
                               int* SIMULATION_TYPE_f, 
                               float* b_deltatover2_F,
                               int* OCEANS){}

void FC_FUNC_(kernel_3_b_cuda,
              KERNEL_3_B_CUDA)(long* Mesh_pointer,
                             int* size_F, 
                             float* deltatover2_F, 
                             int* SIMULATION_TYPE_f, 
			       float* b_deltatover2_F){}

void FC_FUNC_(elastic_ocean_load_cuda,
              ELASTIC_OCEAN_LOAD_CUDA)(long* Mesh_pointer_f, 
                                       int* SIMULATION_TYPE){}

/* from file compute_kernels_cuda.cu */
				      
void FC_FUNC_(compute_kernels_elastic_cuda,
              COMPUTE_KERNELS_ELASTIC_CUDA)(long* Mesh_pointer,
                                            float* deltat){}

void FC_FUNC_(compute_kernels_strength_noise_cuda,
              COMPUTE_KERNELS_STRENGTH_NOISE_CUDA)(long* Mesh_pointer, 
                                                    float* h_noise_surface_movie,
                                                    int* num_free_surface_faces_f,
						   float* deltat){}

void FC_FUNC_(compute_kernels_acoustic_cuda,
              COMPUTE_KERNELS_ACOUSTIC_CUDA)(
                                             long* Mesh_pointer, 
                                             float* deltat){}


/* from file compute_stacey_acoustic_cuda.cu */
void FC_FUNC_(compute_stacey_acoustic_cuda,
              COMPUTE_STACEY_ACOUSTIC_CUDA)(
                                    long* Mesh_pointer_f, 
                                    int* phase_is_innerf, 
                                    int* SIMULATION_TYPEf, 
                                    int* SAVE_FORWARDf,
                                    float* h_b_absorb_potential){}


/* from file compute_stacey_elastic_cuda.cu */
					   
void FC_FUNC_(compute_stacey_elastic_cuda,
              COMPUTE_STACEY_ELASTIC_CUDA)(long* Mesh_pointer_f, 
                                           int* phase_is_innerf, 
                                           int* SIMULATION_TYPEf, 
                                           int* SAVE_FORWARDf,
                                           float* h_b_absorb_field){}

/* from file it_update_displacement_scheme_cuda.cu */
					  
void FC_FUNC_(it_update_displacement_scheme_cuda,
              IT_UPDATE_DISPLACMENT_SCHEME_CUDA)(long* Mesh_pointer_f,
                                                 int* size_F, 
                                                 float* deltat_F, 
                                                 float* deltatsqover2_F, 
                                                 float* deltatover2_F,
                                                 int* SIMULATION_TYPE, 
                                                 float* b_deltat_F, 
                                                 float* b_deltatsqover2_F, 
                                                 float* b_deltatover2_F){}

void FC_FUNC_(it_update_displacement_scheme_acoustic_cuda,
              IT_UPDATE_DISPLACEMENT_SCHEME_ACOUSTIC_CUDA)(long* Mesh_pointer_f, 
                                                           int* size_F,
                                                           float* deltat_F, 
                                                           float* deltatsqover2_F, 
                                                           float* deltatover2_F,
                                                           int* SIMULATION_TYPE, 
                                                           float* b_deltat_F, 
                                                           float* b_deltatsqover2_F, 
                                                           float* b_deltatover2_F){}

/* from file noise_tomography_cuda.cu */
							  
void FC_FUNC_(fortranflush,FORTRANFLUSH)(int* rank){}							  
void FC_FUNC_(fortranprint,FORTRANPRINT)(int* id){}					

void FC_FUNC_(fortranprintf,FORTRANPRINTF)(float* val){}

void FC_FUNC_(fortranprintd,FORTRANPRINTD)(double* val){}

void FC_FUNC_(make_displ_rand,MAKE_DISPL_RAND)(long* Mesh_pointer_f,float* h_displ){}

void FC_FUNC_(transfer_surface_to_host,
              TRANSFER_SURFACE_TO_HOST)(long* Mesh_pointer_f,real* h_noise_surface_movie,int* num_free_surface_faces){}

void FC_FUNC_(noise_read_add_surface_movie_cuda,
              NOISE_READ_ADD_SURFACE_MOVIE_CUDA)(long* Mesh_pointer_f, real* h_noise_surface_movie, int* num_free_surface_faces_f,int* NOISE_TOMOGRAPHYf){}

						
/* from file prepare_mesh_constants_cuda.cu						 */
						
void FC_FUNC_(pause_for_debug,PAUSE_FOR_DEBUG)(){}
void FC_FUNC_(output_free_device_memory,
              OUTPUT_FREE_DEVICE_MEMORY)(int* id){}

void FC_FUNC_(show_free_device_memory,
              SHOW_FREE_DEVICE_MEMORY)(){}

void FC_FUNC_(get_free_device_memory,
              get_FREE_DEVICE_MEMORY)(float* free, float* used, float* total ){}

void FC_FUNC_(prepare_constants_device,
              PREPARE_CONSTANTS_DEVICE)(long* Mesh_pointer,
                                        int* h_NGLLX, 
                                        int* NSPEC_AB, int* NGLOB_AB,
                                        float* h_xix, float* h_xiy, float* h_xiz,
                                        float* h_etax, float* h_etay, float* h_etaz,
                                        float* h_gammax, float* h_gammay, float* h_gammaz,
                                        float* h_kappav, float* h_muv,
                                        int* h_ibool, 
                                        int* num_interfaces_ext_mesh, int* max_nibool_interfaces_ext_mesh,
                                        int* h_nibool_interfaces_ext_mesh, int* h_ibool_interfaces_ext_mesh,
                                        float* h_hprime_xx,float* h_hprime_yy,float* h_hprime_zz, 
                                        float* h_hprimewgll_xx,float* h_hprimewgll_yy,float* h_hprimewgll_zz,
                                        float* h_wgllwgll_xy,float* h_wgllwgll_xz,float* h_wgllwgll_yz,        
                                        int* ABSORBING_CONDITIONS,    
                                        int* h_abs_boundary_ispec, int* h_abs_boundary_ijk,
                                        float* h_abs_boundary_normal,
                                        float* h_abs_boundary_jacobian2Dw,
                                        int* h_num_abs_boundary_faces,
                                        int* h_ispec_is_inner, 
                                        int* NSOURCES,
                                        float* h_sourcearrays,
                                        int* h_islice_selected_source,
                                        int* h_ispec_selected_source,
                                        int* h_number_receiver_global,
                                        int* h_ispec_selected_rec,
                                        int* nrec_f,
                                        int* nrec_local_f,
                                        int* SIMULATION_TYPE)
{
  fprintf(stderr,"ERROR: GPU_MODE enabled without GPU/CUDA Support. To enable GPU support, reconfigure with --with-cuda flag.\n");
  exit(1);
}

void FC_FUNC_(prepare_adjoint_sim2_or_3_constants_device,
              PREPARE_ADJOINT_SIM2_OR_3_CONSTANTS_DEVICE)(
                                                          long* Mesh_pointer_f,
                                                          int* islice_selected_rec,
                                                          int* islice_selected_rec_size){}

void FC_FUNC_(prepare_fields_acoustic_device,
              PREPARE_FIELDS_ACOUSTIC_DEVICE)(long* Mesh_pointer_f, 
                                              float* rmass_acoustic, 
                                              float* rhostore,
                                              float* kappastore,
                                              int* num_phase_ispec_acoustic, 
                                              int* phase_ispec_inner_acoustic,
                                              int* ispec_is_acoustic,
                                              int* NOISE_TOMOGRAPHY,
                                              int* num_free_surface_faces,
                                              int* free_surface_ispec,
                                              int* free_surface_ijk,
                                              int* ABSORBING_CONDITIONS,
                                              int* b_reclen_potential,
                                              float* b_absorb_potential,
                                              int* SIMULATION_TYPE,
                                              float* rho_ac_kl,
                                              float* kappa_ac_kl,
                                              int* ELASTIC_SIMULATION,
                                              int* num_coupling_ac_el_faces,
                                              int* coupling_ac_el_ispec,
                                              int* coupling_ac_el_ijk,
                                              float* coupling_ac_el_normal,
                                              float* coupling_ac_el_jacobian2Dw
                                              ){}							 
void FC_FUNC_(prepare_fields_elastic_device,
              PREPARE_FIELDS_ELASTIC_DEVICE)(long* Mesh_pointer_f, 
                                             int* size,
                                             float* rmass,
                                             float* rho_vp,  
                                             float* rho_vs,
                                             int* num_phase_ispec_elastic, 
                                             int* phase_ispec_inner_elastic,
                                             int* ispec_is_elastic,
                                             int* ABSORBING_CONDITIONS,
                                             float* h_b_absorb_field,
                                             int* h_b_reclen_field,
                                             int* SIMULATION_TYPE,
                                             float* rho_kl,
                                             float* mu_kl,
                                             float* kappa_kl,
                                             int* COMPUTE_AND_STORE_STRAIN,
                                             float* epsilondev_xx,float* epsilondev_yy,float* epsilondev_xy,
                                             float* epsilondev_xz,float* epsilondev_yz,
                                             float* epsilon_trace_over_3,
                                             float* b_epsilondev_xx,float* b_epsilondev_yy,float* b_epsilondev_xy,
                                             float* b_epsilondev_xz,float* b_epsilondev_yz,
                                             float* b_epsilon_trace_over_3,
                                             int* ATTENUATION, int* R_size,
                                             float* R_xx,float* R_yy,float* R_xy,float* R_xz,float* R_yz,
                                             float* b_R_xx,float* b_R_yy,float* b_R_xy,float* b_R_xz,float* b_R_yz,
                                             float* one_minus_sum_beta,float* factor_common,
                                             float* alphaval,float* betaval,float* gammaval,
                                             float* b_alphaval,float* b_betaval,float* b_gammaval,
                                             int* OCEANS,float* rmass_ocean_load,
                                             float* free_surface_normal,int* num_free_surface_faces){}

void FC_FUNC_(prepare_fields_noise_device,
              PREPARE_FIELDS_NOISE_DEVICE)(long* Mesh_pointer_f, 
                                           int* NSPEC_AB, int* NGLOB_AB,
                                           int* free_surface_ispec,int* free_surface_ijk,
                                           int* num_free_surface_faces,
                                           int* size_free_surface_ijk, 
                                           int* SIMULATION_TYPE,
                                           int* NOISE_TOMOGRAPHY,
                                           int* NSTEP,
                                           float* noise_sourcearray,
                                           float* normal_x_noise,
                                           float* normal_y_noise,
                                           float* normal_z_noise,
                                           float* mask_noise,
                                           float* free_surface_jacobian2Dw,
                                           float* Sigma_kl
                                           ){}

void FC_FUNC_(prepare_cleanup_device,
              PREPARE_CLEANUP_DEVICE)(long* Mesh_pointer_f,
                                      int* SIMULATION_TYPE,
                                      int* ACOUSTIC_SIMULATION,
                                      int* ELASTIC_SIMULATION,
                                      int* ABSORBING_CONDITIONS,
                                      int* NOISE_TOMOGRAPHY,
                                      int* COMPUTE_AND_STORE_STRAIN,
                                      int* ATTENUATION){}

/* from file transfer_fields_cuda.cu				      */

void FC_FUNC_(transfer_b_fields_to_device,
              TRANSFER_B_FIELDS_TO_DEVICE)(int* size, float* b_displ, float* b_veloc, float* b_accel,
                                           long* Mesh_pointer_f){}

void FC_FUNC_(transfer_fields_to_device,
              TRANSFER_FIELDS_TO_DEVICE)(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_b_fields_from_device,
              TRANSFER_B_FIELDS_FROM_DEVICE)(int* size, float* b_displ, float* b_veloc, float* b_accel,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_fields_from_device,
              TRANSFER_FIELDS_FROM_DEVICE)(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_accel_to_device,
              TRNASFER_ACCEL_TO_DEVICE)(int* size, float* accel,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_accel_from_device,
              TRANSFER_ACCEL_FROM_DEVICE)(int* size, float* accel,long* Mesh_pointer_f){}
void FC_FUNC_(transfer_b_accel_from_device,
              TRNASFER_B_ACCEL_FROM_DEVICE)(int* size, float* b_accel,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_sigma_from_device,
              TRANSFER_SIGMA_FROM_DEVICE)(int* size, float* sigma_kl,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_b_displ_from_device,
              TRANSFER_B_DISPL_FROM_DEVICE)(int* size, float* displ,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_displ_from_device,
              TRANSFER_DISPL_FROM_DEVICE)(int* size, float* displ,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_compute_kernel_answers_from_device,
              TRANSFER_COMPUTE_KERNEL_ANSWERS_FROM_DEVICE)(long* Mesh_pointer,
                                                           float* rho_kl,int* size_rho,
                                                           float* mu_kl, int* size_mu,
                                                           float* kappa_kl, int* size_kappa){}

void FC_FUNC_(transfer_b_fields_att_to_device,
              TRANSFER_B_FIELDS_ATT_TO_DEVICE)(long* Mesh_pointer,
                                             float* b_R_xx,float* b_R_yy,float* b_R_xy,float* b_R_xz,float* b_R_yz,
                                             int* size_R,
                                             float* b_epsilondev_xx,
                                             float* b_epsilondev_yy,
                                             float* b_epsilondev_xy,
                                             float* b_epsilondev_xz,
                                             float* b_epsilondev_yz,
                                             int* size_epsilondev){}

void FC_FUNC_(transfer_fields_att_from_device,
              TRANSFER_FIELDS_ATT_FROM_DEVICE)(long* Mesh_pointer,
                                               float* R_xx,float* R_yy,float* R_xy,float* R_xz,float* R_yz,
                                               int* size_R,
                                               float* epsilondev_xx,
                                               float* epsilondev_yy,
                                               float* epsilondev_xy,
                                               float* epsilondev_xz,
                                               float* epsilondev_yz,
                                               int* size_epsilondev){}

void FC_FUNC_(transfer_sensitivity_kernels_to_host,
              TRANSFER_SENSITIVITY_KERNELS_TO_HOST)(long* Mesh_pointer, 
                                                    float* h_rho_kl,
                                                    float* h_mu_kl, 
                                                    float* h_kappa_kl,
                                                    int* NSPEC_AB){}

void FC_FUNC_(transfer_sensitivity_kernels_noise_to_host,
              TRANSFER_SENSITIVITY_KERNELS_NOISE_TO_HOST)(long* Mesh_pointer, 
                                                          float* h_Sigma_kl,
                                                          int* NSPEC_AB){}

							 
void FC_FUNC_(transfer_fields_acoustic_to_device,
              TRANSFER_FIELDS_ACOUSTIC_TO_DEVICE)(
                                                  int* size, 
                                                  float* potential_acoustic, 
                                                  float* potential_dot_acoustic, 
                                                  float* potential_dot_dot_acoustic,
                                                  long* Mesh_pointer_f){}

void FC_FUNC_(transfer_b_fields_acoustic_to_device,
              TRANSFER_B_FIELDS_ACOUSTIC_TO_DEVICE)(
                                                    int* size, 
                                                    float* b_potential_acoustic, 
                                                    float* b_potential_dot_acoustic, 
                                                    float* b_potential_dot_dot_acoustic,
                                                    long* Mesh_pointer_f){}

void FC_FUNC_(transfer_fields_acoustic_from_device,TRANSFER_FIELDS_ACOUSTIC_FROM_DEVICE)(
                                                                                         int* size, 
                                                                                         float* potential_acoustic, 
                                                                                         float* potential_dot_acoustic, 
                                                                                         float* potential_dot_dot_acoustic,
                                                                                         long* Mesh_pointer_f){}

void FC_FUNC_(transfer_b_fields_acoustic_from_device,
              TRANSFER_B_FIELDS_ACOUSTIC_FROM_DEVICE)(
                                                      int* size, 
                                                      float* b_potential_acoustic, 
                                                      float* b_potential_dot_acoustic, 
                                                      float* b_potential_dot_dot_acoustic,
                                                      long* Mesh_pointer_f){}

void FC_FUNC_(transfer_potential_dot_dot_from_device,
              TRNASFER_B_ACCEL_FROM_DEVICE)(int* size, float* potential_dot_dot_acoustic,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_b_potential_dot_dot_from_device,
              TRNASFER_B_ACCEL_FROM_DEVICE)(int* size, float* b_potential_dot_dot_acoustic,long* Mesh_pointer_f){}

void FC_FUNC_(transfer_sensitivity_kernels_acoustic_to_host,
              TRANSFER_SENSITIVITY_KERNELS_ACOUSTIC_TO_HOST)(long* Mesh_pointer, 
                                                             float* h_rho_ac_kl,
                                                             float* h_kappa_ac_kl,
                                                             int* NSPEC_AB){}

/* from file write_seismograms_cuda.cu */

void FC_FUNC_(transfer_station_fields_from_device,
              TRANSFER_STATION_FIELDS_FROM_DEVICE)(float* displ,float* veloc,float* accel,
                                                   float* b_displ, float* b_veloc, float* b_accel,
                                                   long* Mesh_pointer_f,int* number_receiver_global,
                                                   int* ispec_selected_rec,int* ispec_selected_source,
                                                   int* ibool,int* SIMULATION_TYPEf){}

void FC_FUNC_(transfer_station_fields_acoustic_from_device,
              TRANSFER_STATION_FIELDS_ACOUSTIC_FROM_DEVICE)(
                                                            float* potential_acoustic,
                                                            float* potential_dot_acoustic,
                                                            float* potential_dot_dot_acoustic,
                                                            float* b_potential_acoustic, 
                                                            float* b_potential_dot_acoustic, 
                                                            float* b_potential_dot_dot_acoustic,
                                                            long* Mesh_pointer_f,
                                                            int* number_receiver_global,
                                                            int* ispec_selected_rec,
                                                            int* ispec_selected_source,
                                                            int* ibool,
                                                            int* SIMULATION_TYPEf){}

							   
							    
