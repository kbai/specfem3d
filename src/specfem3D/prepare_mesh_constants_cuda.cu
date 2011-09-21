#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "mesh_constants_cuda.h"

#include "prepare_constants_cuda.h"

#define MAX(x,y)                    (((x) < (y)) ? (y) : (x))

typedef float real;
 
extern "C" void pause_for_debug_() {
  pause_for_debugger(1);
}

void pause_for_debugger(int pause) {
  if(pause) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);  
    printf("I'm rank %d\n",myrank);
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s:%d ready for attach\n", getpid(), hostname,myrank);
    FILE *file = fopen("/scratch/eiger/rietmann/attach_gdb.txt","w+");
    fprintf(file,"PID %d on %s:%d ready for attach\n", getpid(), hostname,myrank);
    fclose(file);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
}

void exit_on_cuda_error(char* kernel_name) {

cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {      
      fprintf(stderr,"Error after %s: %s\n", kernel_name, cudaGetErrorString(err));
      pause_for_debugger(0);
      exit(1);
    }
}

// Saves GPU memory usage to file
void output_free_memory(char* info_str) {
  int proc;
  MPI_Comm_rank(MPI_COMM_WORLD,&proc);  
  FILE* fp;
  char filename[BUFSIZ];
  sprintf(filename,"../in_out_files/OUTPUT_FILES/gpu_mem_usage_proc_%03d.txt",proc);
  fp = fopen(filename,"a+");

  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status ){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(1); 
  }
 
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  fprintf(fp,"%d: @%s GPU memory usage: used = %f, free = %f MB, total = %f MB\n", proc, info_str,
	 used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

// Fortran-callable version of above method
extern "C" void output_free_memory_(int* id) {
  char info[6];
  sprintf(info,"f %d:",id);
  output_free_memory(info);
}

void show_free_memory(char* info_str) {

  // show memory usage of GPU
  int proc;
  MPI_Comm_rank(MPI_COMM_WORLD,&proc);
  
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status ){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(1);
  }

  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  printf("%d: @%s GPU memory usage: used = %f, free = %f MB, total = %f MB\n", proc, info_str,
	 used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  
}

extern "C" void show_free_device_memory_() {
  show_free_memory("from fortran");
}

void prepare_constants(int NGLLX, int NSPEC_AB, int NGLOB_AB,
		       float* h_xix, float* h_xiy, float* h_xiz,
		       float** d_xix, float** d_xiy, float** d_xiz,
		       float* h_etax, float* h_etay, float* h_etaz,
		       float** d_etax, float** d_etay, float** d_etaz,
		       float* h_gammax, float* h_gammay, float* h_gammaz,
		       float** d_gammax, float** d_gammay, float** d_gammaz,
		       float* h_kappav, float* h_muv,
		       float** d_kappav, float** d_muv,
		       int* h_ibool, int** d_ibool,
		       int* h_phase_ispec_inner_elastic, int** d_phase_ispec_inner_elastic,
		       int num_phase_ispec_elastic,
		       float* h_rmass, float** d_rmass,
		       int num_interfaces_ext_mesh, int max_nibool_interfaces_ext_mesh,
		       int* h_nibool_interfaces_ext_mesh, int** d_nibool_interfaces_ext_mesh,
		       int* h_ibool_interfaces_ext_mesh, int** d_ibool_interfaces_ext_mesh,		       
		       float* h_hprime_xx, float* h_hprimewgll_xx,
		       float* h_wgllwgll_xy, float* h_wgllwgll_xz,
		       float* h_wgllwgll_yz,
		       int* h_abs_boundary_ispec, int** d_abs_boundary_ispec,
		       int* h_abs_boundary_ijk, int** d_abs_boundary_ijk,
		       float* h_abs_boundary_normal, float** d_abs_boundary_normal,
		       float* h_rho_vp,float** d_rho_vp,
		       float* h_rho_vs,float** d_rho_vs,
		       float* h_abs_boundary_jacobian2Dw,float** d_abs_boundary_jacobian2Dw,
		       float* h_b_absorb_field,float** d_b_absorb_field,
		       int num_abs_boundary_faces, int b_num_abs_boundary_faces,
		       int* h_ispec_is_inner, int** d_ispec_is_inner,
		       int* h_ispec_is_elastic, int** d_ispec_is_elastic,
		       int NSOURCES,
		       float* h_sourcearrays,float** d_sourcearrays,
		       int* h_islice_selected_source, int** d_islice_selected_source,
		       int* h_ispec_selected_source, int** d_ispec_selected_source
		       )
{
  
  // EPIK_USER_REG(r_name,"compute_forces");
  // EPIK_USER_REG(r_name,
  
  /* Assuming NGLLX=5. Padded is then 128 (5^3+3) */
  int size_padded = 128*NSPEC_AB;
  int size = NGLLX*NGLLX*NGLLX*NSPEC_AB;
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_xix, size_padded*sizeof(float)),5);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_xiy, size_padded*sizeof(float)),6);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_xiz, size_padded*sizeof(float)),7);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_etax, size_padded*sizeof(float)),8);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_etay, size_padded*sizeof(float)),9);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_etaz, size_padded*sizeof(float)),10);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_gammax, size_padded*sizeof(float)),11);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_gammay, size_padded*sizeof(float)),12);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_gammaz, size_padded*sizeof(float)),13);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_kappav, size_padded*sizeof(float)),14); 
  print_CUDA_error_if_any(cudaMalloc((void**) d_muv, size_padded*sizeof(float)),15);	 
  print_CUDA_error_if_any(cudaMalloc((void**) d_ibool, size_padded*sizeof(int)),16);
  print_CUDA_error_if_any(cudaMalloc((void**) d_phase_ispec_inner_elastic, num_phase_ispec_elastic*2*sizeof(int)),17);
  print_CUDA_error_if_any(cudaMalloc((void**) d_rmass, NGLOB_AB*sizeof(float)),17);

  print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_ispec,
				     num_abs_boundary_faces*sizeof(int)),69);
  print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_ispec, h_abs_boundary_ispec,
				     num_abs_boundary_faces*sizeof(int),
				     cudaMemcpyHostToDevice),70);
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_ijk,
				     3*25*num_abs_boundary_faces*sizeof(int)),2);
  print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_ijk, h_abs_boundary_ijk,
				     3*25*num_abs_boundary_faces*sizeof(int),
				     cudaMemcpyHostToDevice),2);
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_normal,
				     3*25*num_abs_boundary_faces*sizeof(int)),3);
  print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_normal, h_abs_boundary_normal,
				     3*25*num_abs_boundary_faces*sizeof(int),
				     cudaMemcpyHostToDevice),3);
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_jacobian2Dw,
				     25*num_abs_boundary_faces*sizeof(float)),4);
  print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_jacobian2Dw, h_abs_boundary_jacobian2Dw,
				     25*num_abs_boundary_faces*sizeof(float),
				     cudaMemcpyHostToDevice),1);
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_rho_vp, size*sizeof(float)),5);
  print_CUDA_error_if_any(cudaMalloc((void**) d_rho_vs, size*sizeof(float)),6);
  print_CUDA_error_if_any(cudaMemcpy(*d_rho_vp,h_rho_vp,size*sizeof(float),
				     cudaMemcpyHostToDevice),5);
  print_CUDA_error_if_any(cudaMemcpy(*d_rho_vs,h_rho_vs,size*sizeof(float),
				     cudaMemcpyHostToDevice),5);
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_b_absorb_field, 3*25*b_num_abs_boundary_faces*sizeof(float)),7);
  print_CUDA_error_if_any(cudaMemcpy(*d_b_absorb_field, h_b_absorb_field,
				     3*25*b_num_abs_boundary_faces*sizeof(float),
				     cudaMemcpyHostToDevice),7);
  
  print_CUDA_error_if_any(cudaMemcpy(*d_rmass,h_rmass,NGLOB_AB*sizeof(float),cudaMemcpyHostToDevice),18);

  // prepare interprocess-edge exchange information
  print_CUDA_error_if_any(cudaMalloc((void**) d_nibool_interfaces_ext_mesh,
				     num_interfaces_ext_mesh*sizeof(int)),19);
  print_CUDA_error_if_any(cudaMemcpy(*d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
				     num_interfaces_ext_mesh*sizeof(int),cudaMemcpyHostToDevice),19);
  
  print_CUDA_error_if_any(cudaMalloc((void**) d_ibool_interfaces_ext_mesh,
				     num_interfaces_ext_mesh*max_nibool_interfaces_ext_mesh*sizeof(int)),20);
  print_CUDA_error_if_any(cudaMemcpy(*d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
				     num_interfaces_ext_mesh*max_nibool_interfaces_ext_mesh*sizeof(int),
				     cudaMemcpyHostToDevice),20);

  print_CUDA_error_if_any(cudaMalloc((void**) d_ispec_is_inner,NSPEC_AB*sizeof(int)),21);
  print_CUDA_error_if_any(cudaMemcpy(*d_ispec_is_inner, h_ispec_is_inner,
				     NSPEC_AB*sizeof(int),
				     cudaMemcpyHostToDevice),21);
  print_CUDA_error_if_any(cudaMalloc((void**) d_ispec_is_elastic,NSPEC_AB*sizeof(int)),21);
  print_CUDA_error_if_any(cudaMemcpy(*d_ispec_is_elastic, h_ispec_is_elastic,
				     NSPEC_AB*sizeof(int),
				     cudaMemcpyHostToDevice),21);

  print_CUDA_error_if_any(cudaMemcpy(*d_ibool, h_ibool,
				     size*sizeof(int)  ,cudaMemcpyHostToDevice),12);    

  print_CUDA_error_if_any(cudaMalloc((void**)d_sourcearrays, sizeof(float)*NSOURCES*3*125),22);
  print_CUDA_error_if_any(cudaMemcpy(*d_sourcearrays, h_sourcearrays, sizeof(float)*NSOURCES*3*125,
				     cudaMemcpyHostToDevice),22);

  print_CUDA_error_if_any(cudaMalloc((void**)d_islice_selected_source, sizeof(int)*NSOURCES),23);
  print_CUDA_error_if_any(cudaMemcpy(*d_islice_selected_source, h_islice_selected_source, sizeof(int)*NSOURCES,
				     cudaMemcpyHostToDevice),23);

  print_CUDA_error_if_any(cudaMalloc((void**)d_ispec_selected_source, sizeof(int)*NSOURCES),24);
  print_CUDA_error_if_any(cudaMemcpy(*d_ispec_selected_source, h_ispec_selected_source,sizeof(int)*NSOURCES,
				     cudaMemcpyHostToDevice),24);
  
  // transfer constant element data with padding
  for(int i=0;i<NSPEC_AB;i++) {
    print_CUDA_error_if_any(cudaMemcpy(*d_xix + i*128, &h_xix[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),69);
    print_CUDA_error_if_any(cudaMemcpy(*d_xiy+i*128,   &h_xiy[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),11);
    print_CUDA_error_if_any(cudaMemcpy(*d_xiz+i*128,   &h_xiz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),3);
    print_CUDA_error_if_any(cudaMemcpy(*d_etax+i*128,  &h_etax[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),4);
    print_CUDA_error_if_any(cudaMemcpy(*d_etay+i*128,  &h_etay[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),5);
    print_CUDA_error_if_any(cudaMemcpy(*d_etaz+i*128,  &h_etaz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),6);
    print_CUDA_error_if_any(cudaMemcpy(*d_gammax+i*128,&h_gammax[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),7);
    print_CUDA_error_if_any(cudaMemcpy(*d_gammay+i*128,&h_gammay[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),8);
    print_CUDA_error_if_any(cudaMemcpy(*d_gammaz+i*128,&h_gammaz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),9);
    print_CUDA_error_if_any(cudaMemcpy(*d_kappav+i*128,&h_kappav[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),10);
    print_CUDA_error_if_any(cudaMemcpy(*d_muv+i*128,   &h_muv[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),11);
      
  }

  
  
  print_CUDA_error_if_any(cudaMemcpy(*d_phase_ispec_inner_elastic, h_phase_ispec_inner_elastic, num_phase_ispec_elastic*2*sizeof(int),cudaMemcpyHostToDevice),13);
  
  

  
  
}


extern "C" void prepare_constants_device_(long* Mesh_pointer,int* NGLLX, int* NSPEC_AB, int* NGLOB_AB,
					  float* h_xix, float* h_xiy, float* h_xiz,
					  float* h_etax, float* h_etay, float* h_etaz,
					  float* h_gammax, float* h_gammay, float* h_gammaz,
					  float* h_kappav, float* h_muv,
					  int* h_ibool, int* h_phase_ispec_inner_elastic,
					  int* num_phase_ispec_elastic,
					  float* h_rmass,
					  int* num_interfaces_ext_mesh, int* max_nibool_interfaces_ext_mesh,
					  int* h_nibool_interfaces_ext_mesh, int* h_ibool_interfaces_ext_mesh,
					  float* h_hprime_xx, float* h_hprimewgll_xx,
					  float* h_wgllwgll_xy, float* h_wgllwgll_xz,
					  float* h_wgllwgll_yz,
					  int* h_abs_boundary_ispec, int* h_abs_boundary_ijk,
					  float* h_abs_boundary_normal,
					  float* h_rho_vp,
					  float* h_rho_vs,
					  float* h_abs_boundary_jacobian2Dw,
					  float* h_b_absorb_field,
					  int* num_abs_boundary_faces, int* b_num_abs_boundary_faces,
					  int* h_ispec_is_inner, int* h_ispec_is_elastic,
					  int* NSOURCES,
					  float* h_sourcearrays,
					  int* h_islice_selected_source,
					  int* h_ispec_selected_source,
					  int* h_number_receiver_global,
					  int* h_ispec_selected_rec,
					  int* nrec_local_f,
					  int* nrec_f
					  ) {
  
  int device_count,procid;
  cuInit(0);
  cudaGetDeviceCount(&device_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  if(device_count > 1) {
    cudaSetDevice((procid)%2);
    exit_on_cuda_error("cudaSetDevice");   
  }

  printf("GPU_MODE Active. Preparing Fields and Constants on Device.\n");
  
  Mesh* mp = (Mesh*)malloc(sizeof(Mesh));
  *Mesh_pointer = (long)mp;
  
  mp->NGLLX = *NGLLX;
  mp->NSPEC_AB = *NSPEC_AB;
  mp->NGLOB_AB = *NGLOB_AB;
  mp->d_num_phase_ispec_elastic = *num_phase_ispec_elastic;
  setConst_hprime_xx    (h_hprime_xx    );
  setConst_hprimewgll_xx(h_hprimewgll_xx);
  setConst_wgllwgll_xy  (h_wgllwgll_xy,mp);
  setConst_wgllwgll_xz  (h_wgllwgll_xz,mp);
  setConst_wgllwgll_yz  (h_wgllwgll_yz,mp);

  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_stf_pre_compute),
				     *NSOURCES*sizeof(double)),1);

  int size_padded = 128* *NSPEC_AB;
  int size = *NGLLX * *NGLLX * *NGLLX * *NSPEC_AB;
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xix, size_padded*sizeof(float)),5);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xiy, size_padded*sizeof(float)),6);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xiz, size_padded*sizeof(float)),7);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etax, size_padded*sizeof(float)),8);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etay, size_padded*sizeof(float)),9);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etaz, size_padded*sizeof(float)),10);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammax, size_padded*sizeof(float)),11);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammay, size_padded*sizeof(float)),12);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammaz, size_padded*sizeof(float)),13);	 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_kappav, size_padded*sizeof(float)),14); 
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_muv, size_padded*sizeof(float)),15);	 

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool, size_padded*sizeof(int)),16);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh* *max_nibool_interfaces_ext_mesh*
				     sizeof(int)),20);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_phase_ispec_inner_elastic, *num_phase_ispec_elastic*2*sizeof(int)),17);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_phase_ispec_inner_elastic, h_phase_ispec_inner_elastic, *num_phase_ispec_elastic*2*sizeof(int),cudaMemcpyHostToDevice),13);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_rmass, *NGLOB_AB*sizeof(float)),17);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass,h_rmass,*NGLOB_AB*sizeof(float),cudaMemcpyHostToDevice),18);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_ispec,
				     *num_abs_boundary_faces*sizeof(int)),69);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ispec, h_abs_boundary_ispec,
				     *num_abs_boundary_faces*sizeof(int),
				     cudaMemcpyHostToDevice),70);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_ijk,
				     3*25**num_abs_boundary_faces*sizeof(int)),2);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ijk, h_abs_boundary_ijk,
				     3*25**num_abs_boundary_faces*sizeof(int),
				     cudaMemcpyHostToDevice),2);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_normal,
				     3*25**num_abs_boundary_faces*sizeof(int)),3);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_normal, h_abs_boundary_normal,
				     3*25**num_abs_boundary_faces*sizeof(int),
				     cudaMemcpyHostToDevice),3);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_jacobian2Dw,
				     25**num_abs_boundary_faces*sizeof(float)),4);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_jacobian2Dw, h_abs_boundary_jacobian2Dw,
				     25**num_abs_boundary_faces*sizeof(float),
				     cudaMemcpyHostToDevice),1);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_rho_vp, size*sizeof(float)),5);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_rho_vs, size*sizeof(float)),6);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vp,h_rho_vp,size*sizeof(float),
				     cudaMemcpyHostToDevice),5);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vs,h_rho_vs,size*sizeof(float),
				     cudaMemcpyHostToDevice),5);
  
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_b_absorb_field, 3*25* *b_num_abs_boundary_faces*sizeof(float)),7);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_b_absorb_field, h_b_absorb_field,
				     3*25* *b_num_abs_boundary_faces*sizeof(float),
				     cudaMemcpyHostToDevice),7);
  

  // prepare interprocess-edge exchange information
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_nibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh*sizeof(int)),19);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh*sizeof(int),cudaMemcpyHostToDevice),19);
  
  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh* *max_nibool_interfaces_ext_mesh*sizeof(int),
				     cudaMemcpyHostToDevice),20);

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ispec_is_inner,*NSPEC_AB*sizeof(int)),21);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_inner, h_ispec_is_inner,
				     *NSPEC_AB*sizeof(int),
				     cudaMemcpyHostToDevice),21);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ispec_is_elastic,*NSPEC_AB*sizeof(int)),21);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_elastic, h_ispec_is_elastic,
				     *NSPEC_AB*sizeof(int),
				     cudaMemcpyHostToDevice),21);

  print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool, h_ibool,
				     size*sizeof(int)  ,cudaMemcpyHostToDevice),12);    

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_sourcearrays, sizeof(float)* *NSOURCES*3*125),22);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_sourcearrays, h_sourcearrays, sizeof(float)* *NSOURCES*3*125,
				     cudaMemcpyHostToDevice),22);

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_source, sizeof(int) * *NSOURCES),23);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_islice_selected_source, h_islice_selected_source, sizeof(int)* *NSOURCES,
				     cudaMemcpyHostToDevice),23);

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_ispec_selected_source, sizeof(int)* *NSOURCES),24);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_selected_source, h_ispec_selected_source,sizeof(int)* *NSOURCES,
				     cudaMemcpyHostToDevice),24);
  
  // transfer constant element data with padding
  for(int i=0;i<*NSPEC_AB;i++) {
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xix + i*128, &h_xix[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),69);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiy+i*128,   &h_xiy[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),11);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiz+i*128,   &h_xiz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),3);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etax+i*128,  &h_etax[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),4);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etay+i*128,  &h_etay[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),5);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etaz+i*128,  &h_etaz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),6);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammax+i*128,&h_gammax[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),7);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammay+i*128,&h_gammay[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),8);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammaz+i*128,&h_gammaz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),9);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_kappav+i*128,&h_kappav[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),10);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_muv+i*128,   &h_muv[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),11);
      
  }
        
  int nrec_local = *nrec_local_f;
  int nrec = *nrec_f;

  // note that:
  // size(number_receiver_global) = nrec_local
  // size(ispec_selected_rec) = nrec
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_number_receiver_global),nrec_local*sizeof(int)),1);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_selected_rec),nrec*sizeof(int)),2);
  cudaMemcpy(mp->d_number_receiver_global,h_number_receiver_global,nrec_local*sizeof(int),
	     cudaMemcpyHostToDevice);
  
  cudaMemcpy(mp->d_ispec_selected_rec,h_ispec_selected_rec,nrec*sizeof(int),
	     cudaMemcpyHostToDevice);
  
  mp->nrec_local = nrec_local;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_station_seismo_field),3*125*nrec_local*sizeof(float)),3);
  mp->h_station_seismo_field = (float*)malloc(3*125*nrec_local*sizeof(float));
  
}

extern "C" void prepare_and_transfer_noise_backward_fields_(long* Mesh_pointer_f,
							    int* size,
							    real* b_displ,
							    real* b_veloc,
							    real* b_accel,
							    real* b_epsilondev_xx,
							    real* b_epsilondev_yy,
							    real* b_epsilondev_xy,
							    real* b_epsilondev_xz,
							    real* b_epsilondev_yz,
							    int* NSPEC_STRAIN_ONLY) {
  //show_free_memory("prep_and_xfer_noise_bwd_fields");
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  int epsilondev_size = 128*(*NSPEC_STRAIN_ONLY);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_displ),*size*sizeof(real)),1);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_veloc),*size*sizeof(real)),2);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_accel),*size*sizeof(real)),3);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xx),
				     epsilondev_size*sizeof(real)),4);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_yy),
				     epsilondev_size*sizeof(real)),4);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xy),
				     epsilondev_size*sizeof(real)),4);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xz),
				     epsilondev_size*sizeof(real)),4);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_yz),
				     epsilondev_size*sizeof(real)),4);

  
  cudaMemcpy(mp->d_b_displ,b_displ,*size*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_veloc,b_veloc,*size*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_accel,b_accel,*size*sizeof(real),cudaMemcpyHostToDevice);  

  cudaMemcpy(mp->d_b_epsilondev_xx,b_epsilondev_xx,
	     epsilondev_size*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_yy,b_epsilondev_yy,
	     epsilondev_size*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_xy,b_epsilondev_xy,
	     epsilondev_size*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_xz,b_epsilondev_xz,
	     epsilondev_size*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilondev_yz,b_epsilondev_yz,
	     epsilondev_size*sizeof(real),cudaMemcpyHostToDevice);
  
}
extern "C" 
void prepare_and_transfer_noise_backward_constants_(long* Mesh_pointer_f,
						    float* normal_x_noise,
						    float* normal_y_noise,
						    float* normal_z_noise,
						    float* mask_noise,
						    float* free_surface_jacobian2Dw,
						    int* nfaces_surface_ext_mesh
						    ) {

  //show_free_memory("prep_and_xfer_noise_bwd_constants");
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  int nface_size = 5*5*(*nfaces_surface_ext_mesh);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_x_noise,
				     nface_size*sizeof(float)),1);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_y_noise,
				     nface_size*sizeof(float)),2);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_z_noise,
				     nface_size*sizeof(float)),3);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_mask_noise, nface_size*sizeof(float)),4);

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_free_surface_jacobian2Dw,
				     nface_size*sizeof(float)),5);

  cudaMemcpy(mp->d_normal_x_noise, normal_x_noise, nface_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_normal_y_noise, normal_y_noise, nface_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_normal_z_noise, normal_z_noise, nface_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_mask_noise, mask_noise, nface_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_free_surface_jacobian2Dw, free_surface_jacobian2Dw, nface_size*sizeof(float),cudaMemcpyHostToDevice);
  printf("jacobian_size = %d\n",25*(*nfaces_surface_ext_mesh));
}

extern "C" void prepare_noise_constants_device_(long* Mesh_pointer_f, int* NGLLX, int* NSPEC_AB, int* NGLOB_AB,
				     int* free_surface_ispec, int* num_free_surface_faces, int* SIMULATION_TYPE) {

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  mp->num_free_surface_faces = *num_free_surface_faces;

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ispec, *num_free_surface_faces*sizeof(int)),1);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec, free_surface_ispec, *num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),1);

  // alloc storage for the surface buffer to be copied
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_noise_surface_movie, 3*25*(*num_free_surface_faces)*sizeof(float)),1);
  
}

extern "C" void prepare_sensitivity_kernels_(long* Mesh_pointer_f,
					     float* rho_kl,
					     float* mu_kl,
					     float* kappa_kl,
					     float* epsilon_trace_over_3,
					     float* b_epsilon_trace_over_3,
					     float* Sigma_kl,
					     int* NSPEC_ADJOINTf) {
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  int NSPEC_ADJOINT = *NSPEC_ADJOINTf;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_kl),
				     125*mp->NSPEC_AB*sizeof(float)),8);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_mu_kl),
				     125*mp->NSPEC_AB*sizeof(float)),8);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_kl),
				     125*mp->NSPEC_AB*sizeof(float)),8);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_epsilon_trace_over_3),
				     125*mp->NSPEC_AB*sizeof(float)),8);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilon_trace_over_3),
				     125*mp->NSPEC_AB*sizeof(float)),8);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_Sigma_kl),
				     125*(mp->NSPEC_AB)*sizeof(float)),9);

  cudaMemcpy(mp->d_rho_kl,rho_kl, 125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_mu_kl,mu_kl, 125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_kappa_kl,kappa_kl, 125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_epsilon_trace_over_3,epsilon_trace_over_3,
	     125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilon_trace_over_3 ,b_epsilon_trace_over_3,
	     125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_Sigma_kl, Sigma_kl, 125*(NSPEC_ADJOINT)*sizeof(float),
	     cudaMemcpyHostToDevice);
  
  exit_on_cuda_error("prepare_sensitivity_kernels");
}
					     

extern "C" void prepare_adjoint_constants_device_(long* Mesh_pointer_f,
						  int* NGLLX,
						  int* ispec_selected_rec,
						  int* islice_selected_rec,
						  int* islice_selected_rec_size,
						  int* nrec,
						  float* noise_sourcearray,
						  int* NSTEP,
						  float* epsilondev_xx,
						  float* epsilondev_yy,
						  float* epsilondev_xy,
						  float* epsilondev_xz,
						  float* epsilondev_yz,
						  int* NSPEC_STRAIN_ONLY
						  ) {
  exit_on_cuda_error("prepare_adjoint_constants_device 1");  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  int epsilondev_size = 128*(*NSPEC_STRAIN_ONLY);
  
  // already done earlier
  // print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_ispec_selected_rec,
  // *nrec*sizeof(int)),1);
  // cudaMemcpy(mp->d_ispec_selected_rec,ispec_selected_rec, *nrec*sizeof(int),  
  // cudaMemcpyHostToDevice);

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_rec,
				     *islice_selected_rec_size*sizeof(int)),2);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_noise_sourcearray,
				     3*125*(*NSTEP)*sizeof(float)),2);

  
  cudaMemcpy(mp->d_noise_sourcearray, noise_sourcearray,
	     3*125*(*NSTEP)*sizeof(float),
	     cudaMemcpyHostToDevice);

  exit_on_cuda_error("prepare_adjoint_constants_device 2");  
  
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xx,
				     epsilondev_size*sizeof(float)),3);
  cudaMemcpy(mp->d_epsilondev_xx,epsilondev_xx,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yy,
				     epsilondev_size*sizeof(float)),4);
  cudaMemcpy(mp->d_epsilondev_yy,epsilondev_yy,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xy,
				     epsilondev_size*sizeof(float)),5);
  cudaMemcpy(mp->d_epsilondev_xy,epsilondev_xy,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xz,
				     epsilondev_size*sizeof(float)),6);
  cudaMemcpy(mp->d_epsilondev_xz,epsilondev_xz,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yz,
				     epsilondev_size*sizeof(float)),7);
  cudaMemcpy(mp->d_epsilondev_yz,epsilondev_yz,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  exit_on_cuda_error("prepare_adjoint_constants_device 3");  
  
  
  // these don't seem necessary and crash code for NOISE_TOMOGRAPHY >
  // 0 b/c rho_kl, etc not yet allocated when NT=1
  
    
}

extern "C" {
  void prepare_fields_device_(long* Mesh_pointer_f, int* size);
  void transfer_fields_to_device_(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f);
  void transfer_fields_from_device_(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f);
}

void prepare_fields_device_(long* Mesh_pointer_f, int* size) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_displ),sizeof(float)*(*size)),0);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_veloc),sizeof(float)*(*size)),1);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_accel),sizeof(float)*(*size)),2);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_accel_buffer),sizeof(float)*(*size)),2);

}


extern "C" void transfer_b_fields_to_device_(int* size, float* b_displ, float* b_veloc, float* b_accel,
					     long* Mesh_pointer_f) {
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  cudaMemcpy(mp->d_b_displ,b_displ,sizeof(float)*(*size),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_veloc,b_veloc,sizeof(float)*(*size),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_accel,b_accel,sizeof(float)*(*size),cudaMemcpyHostToDevice);
  
}

void transfer_fields_to_device_(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f) {
    
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container

  print_CUDA_error_if_any(cudaMemcpy(mp->d_displ,displ,sizeof(float)*(*size),cudaMemcpyHostToDevice),3);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_veloc,veloc,sizeof(float)*(*size),cudaMemcpyHostToDevice),4);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_accel,accel,sizeof(float)*(*size),cudaMemcpyHostToDevice),5);
  
}

extern "C" void transfer_b_fields_from_device_(int* size, float* b_displ, float* b_veloc, float* b_accel,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  cudaMemcpy(b_displ,mp->d_b_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_veloc,mp->d_b_veloc,sizeof(float)*(*size),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_accel,mp->d_b_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost);
  
}

extern "C" void get_max_accel_(int* itf,int* sizef,long* Mesh_pointer) {  
  Mesh* mp = (Mesh*)(*Mesh_pointer);
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  int size = *sizef;
  int it = *itf;
  float* accel_cpy = (float*)malloc(size*sizeof(float));
  cudaMemcpy(accel_cpy,mp->d_accel,size*sizeof(float),cudaMemcpyDeviceToHost);
  float maxval=0;
  for(int i=0;i<size;++i) {
    maxval = MAX(maxval,accel_cpy[i]);
  }
  printf("%d/%d: max=%e\n",it,procid,maxval);
  free(accel_cpy);
}

extern "C" void transfer_accel_to_device_(int* size, float* accel,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_accel,accel,sizeof(float)*(*size),cudaMemcpyHostToDevice),6);

}

extern "C" void transfer_accel_from_device_(int* size, float* accel,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(accel,mp->d_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

extern "C" void transfer_b_accel_from_device_(int* size, float* b_accel,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(b_accel,mp->d_b_accel,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}


extern "C" void transfer_sigma_from_device_(int* size, float* sigma_kl,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(sigma_kl,mp->d_Sigma_kl,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}




extern "C" void transfer_b_displ_from_device_(int* size, float* displ,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

extern "C" void transfer_displ_from_device_(int* size, float* displ,long* Mesh_pointer_f) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  
  print_CUDA_error_if_any(cudaMemcpy(displ,mp->d_displ,sizeof(float)*(*size),cudaMemcpyDeviceToHost),6);

}

extern "C" void transfer_compute_kernel_answers_from_device_(long* Mesh_pointer,
							     float* rho_kl,int* size_rho,
							     float* mu_kl, int* size_mu,
							     float* kappa_kl, int* size_kappa) {
  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  cudaMemcpy(rho_kl,mp->d_rho_kl,*size_rho*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(mu_kl,mp->d_mu_kl,*size_mu*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(kappa_kl,mp->d_kappa_kl,*size_kappa*sizeof(float),cudaMemcpyDeviceToHost);  
  
}

extern "C" void transfer_compute_kernel_fields_from_device_(long* Mesh_pointer,
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
  exit_on_cuda_error("after transfer_compute_kernel_fields_from_device");
}
							    

void transfer_fields_from_device_(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f) {
  
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


extern "C" void check_max_norm_displ_gpu_(int* size, float* displ,long* Mesh_pointer_f,int* announceID) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container  

  cudaMemcpy(displ, mp->d_displ,*size*sizeof(float),cudaMemcpyDeviceToHost);
  float maxnorm=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(displ[i]));
  }
  printf("%d: maxnorm of forward displ = %e\n",*announceID,maxnorm);
}

extern "C" void check_max_norm_vector_(int* size, float* vector1, int* announceID) {
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  float maxnorm=0;
  int maxloc;
  for(int i=0;i<*size;i++) {
    if(maxnorm<fabsf(vector1[i])) {
      maxnorm = vector1[i];
      maxloc = i;
    }
  }
  printf("%d:maxnorm of vector %d [%d] = %e\n",procid,*announceID,maxloc,maxnorm);
}

extern "C" void check_max_norm_displ_(int* size, float* displ, int* announceID) {
  float maxnorm=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(displ[i]));
  }
  printf("%d: maxnorm of forward displ = %e\n",*announceID,maxnorm);
}


extern "C" void check_max_norm_b_displ_gpu_(int* size, float* b_displ,long* Mesh_pointer_f,int* announceID) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container  

  float* b_accel = (float*)malloc(*size*sizeof(float));
  
  cudaMemcpy(b_displ, mp->d_b_displ,*size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(b_accel, mp->d_b_accel,*size*sizeof(float),cudaMemcpyDeviceToHost);

  float maxnorm=0;
  float maxnorm_accel=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(b_displ[i]));
    maxnorm_accel = MAX(maxnorm,fabsf(b_accel[i]));
  }
  free(b_accel);
  printf("%d: maxnorm of backward displ = %e\n",*announceID,maxnorm);
  printf("%d: maxnorm of backward accel = %e\n",*announceID,maxnorm_accel);
}


extern "C" void check_max_norm_b_accel_gpu_(int* size, float* b_accel,long* Mesh_pointer_f,int* announceID) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container  

  cudaMemcpy(b_accel, mp->d_b_accel,*size*sizeof(float),cudaMemcpyDeviceToHost);

  float maxnorm=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(b_accel[i]));
  }
  printf("%d: maxnorm of backward accel = %e\n",*announceID,maxnorm);
}

extern "C" void check_max_norm_b_veloc_gpu_(int* size, float* b_veloc,long* Mesh_pointer_f,int* announceID) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container  

  cudaMemcpy(b_veloc, mp->d_b_veloc,*size*sizeof(float),cudaMemcpyDeviceToHost);

  float maxnorm=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(b_veloc[i]));
  }
  printf("%d: maxnorm of backward veloc = %e\n",*announceID,maxnorm);
}

extern "C" void check_max_norm_b_displ_(int* size, float* b_displ,int* announceID) {
    
  float maxnorm=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(b_displ[i]));
  }
  printf("%d:maxnorm of backward displ = %e\n",*announceID,maxnorm);
}


extern "C" void check_max_norm_b_accel_(int* size, float* b_accel,int* announceID) {
    
  float maxnorm=0;
  
  for(int i=0;i<*size;i++) {
    maxnorm = MAX(maxnorm,fabsf(b_accel[i]));
  }
  printf("%d:maxnorm of backward accel = %e\n",*announceID,maxnorm);
}

extern "C" void check_error_vectors_(int* sizef, float* vector1,float* vector2) {

  int size = *sizef;

  double diff2 = 0;
  double sum = 0;
  double temp;
  double maxerr=0;
  int maxerrorloc;
  
  for(int i=0;i<size;++i) {
    temp = vector1[i]-vector2[i];    
    diff2 += temp*temp;
    sum += vector1[i]*vector1[i];
    if(maxerr < fabsf(temp)) {
      maxerr = abs(temp);
      maxerrorloc = i;
    }
  }

  printf("rel error = %f, maxerr = %e @ %d\n",diff2/sum,maxerr,maxerrorloc); 
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if(myrank==0) {
    for(int i=maxerrorloc;i>maxerrorloc-5;i--) {
      printf("[%d]: %e vs. %e\n",i,vector1[i],vector2[i]);
    }
  }
  
}
