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

// Helper functions

/* ----------------------------------------------------------------------------------------------- */

double get_time()
{
  struct timeval t;
  struct timezone tzp;
  gettimeofday(&t, &tzp);
  return t.tv_sec + t.tv_usec*1e-6;
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(pause_for_debug,PAUSE_FOR_DEBUG)() {
TRACE("pause_for_debug");  

  pause_for_debugger(1);
}

/* ----------------------------------------------------------------------------------------------- */

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

/* ----------------------------------------------------------------------------------------------- */

void exit_on_cuda_error(char* kernel_name) {
  // sync and check to catch errors from previous async operations
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {      
      fprintf(stderr,"Error after %s: %s\n", kernel_name, cudaGetErrorString(err));
      pause_for_debugger(0);
      exit(1);
    }
}

/* ----------------------------------------------------------------------------------------------- */

void print_CUDA_error_if_any(cudaError_t err, int num)
{
  if (cudaSuccess != err)
  {
    printf("\nCUDA error !!!!! <%s> !!!!! at CUDA call # %d\n",cudaGetErrorString(err),num);
    fflush(stdout);
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    exit(0);
  }
  return;
}

/* ----------------------------------------------------------------------------------------------- */

void get_free_memory(double* free_db, double* used_db, double* total_db) {
  
  // gets memory usage in byte
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status ){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(1); 
  }
  
  *free_db = (double)free_byte ;
  *total_db = (double)total_byte ;
  *used_db = *total_db - *free_db ;
  return;  
}

/* ----------------------------------------------------------------------------------------------- */

// Saves GPU memory usage to file
void output_free_memory(char* info_str) {
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);  
  FILE* fp;
  char filename[BUFSIZ];
  double free_db,used_db,total_db;

  get_free_memory(&free_db,&used_db,&total_db);
  //  size_t free_byte ;
  //  size_t total_byte ;
  //  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  //  if ( cudaSuccess != cuda_status ){
  //    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
  //    exit(1); 
  //  }
  // 
  //  double free_db = (double)free_byte ;
  //  double total_db = (double)total_byte ;
  //  double used_db = total_db - free_db ;
  
  sprintf(filename,"../in_out_files/OUTPUT_FILES/gpu_mem_usage_proc_%03d.txt",myrank);
  fp = fopen(filename,"a+");
  fprintf(fp,"%d: @%s GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n", myrank, info_str,
	 used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  fclose(fp);
}

/* ----------------------------------------------------------------------------------------------- */

// Fortran-callable version of above method
extern "C"
void FC_FUNC_(output_free_device_memory,
              OUTPUT_FREE_DEVICE_MEMORY)(int* id) {
TRACE("output_free_device_memory");  
              
  char info[6];
  sprintf(info,"f %d:",*id);
  output_free_memory(info);
}

/* ----------------------------------------------------------------------------------------------- */

void show_free_memory(char* info_str) {

  // show memory usage of GPU
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  double free_db,used_db,total_db;
  
  get_free_memory(&free_db,&used_db,&total_db);
  
//  size_t free_byte ;
//  size_t total_byte ;
//  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
//  if ( cudaSuccess != cuda_status ){
//    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
//    exit(1);
//  }
//
//  double free_db = (double)free_byte ;
//  double total_db = (double)total_byte ;
//  double used_db = total_db - free_db ;

  printf("%d: @%s GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n", myrank, info_str,
	 used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(show_free_device_memory,
              SHOW_FREE_DEVICE_MEMORY)() {
TRACE("show_free_device_memory");  

  show_free_memory("from fortran");
}


extern "C"
void FC_FUNC_(get_free_device_memory,
              get_FREE_DEVICE_MEMORY)(float* free, float* used, float* total ) {
TRACE("get_free_device_memory");  

  double free_db,used_db,total_db;
  
  get_free_memory(&free_db,&used_db,&total_db);
  
  // converts to MB
  *free = (float) free_db/1024.0/1024.0;
  *used = (float) used_db/1024.0/1024.0;
  *total = (float) total_db/1024.0/1024.0;
  return;
}



/* ----------------------------------------------------------------------------------------------- */

// GPU preparation

/* ----------------------------------------------------------------------------------------------- */

//void prepare_constants(int NGLLX, int NSPEC_AB, int NGLOB_AB,
//		       float* h_xix, float* h_xiy, float* h_xiz,
//		       float** d_xix, float** d_xiy, float** d_xiz,
//		       float* h_etax, float* h_etay, float* h_etaz,
//		       float** d_etax, float** d_etay, float** d_etaz,
//		       float* h_gammax, float* h_gammay, float* h_gammaz,
//		       float** d_gammax, float** d_gammay, float** d_gammaz,
//		       float* h_kappav, float* h_muv,
//		       float** d_kappav, float** d_muv,
//		       int* h_ibool, int** d_ibool,
//		       //int* h_phase_ispec_inner_elastic, int** d_phase_ispec_inner_elastic,
//		       //int num_phase_ispec_elastic,
//		       //float* h_rmass, float** d_rmass,
//		       int num_interfaces_ext_mesh, int max_nibool_interfaces_ext_mesh,
//		       int* h_nibool_interfaces_ext_mesh, int** d_nibool_interfaces_ext_mesh,
//		       int* h_ibool_interfaces_ext_mesh, int** d_ibool_interfaces_ext_mesh,		       
//		       float* h_hprime_xx, float* h_hprimewgll_xx,
//		       float* h_wgllwgll_xy, float* h_wgllwgll_xz,
//		       float* h_wgllwgll_yz,
//		       int* h_abs_boundary_ispec, int** d_abs_boundary_ispec,
//		       int* h_abs_boundary_ijk, int** d_abs_boundary_ijk,
//		       float* h_abs_boundary_normal, float** d_abs_boundary_normal,
//		       //float* h_rho_vp,float** d_rho_vp,
//		       //float* h_rho_vs,float** d_rho_vs,
//		       float* h_abs_boundary_jacobian2Dw,float** d_abs_boundary_jacobian2Dw,
//		       float* h_b_absorb_field,float** d_b_absorb_field,
//		       int num_abs_boundary_faces, int b_num_abs_boundary_faces,
//		       int* h_ispec_is_inner, int** d_ispec_is_inner,
//		       //int* h_ispec_is_elastic, int** d_ispec_is_elastic,
//		       int NSOURCES,
//		       float* h_sourcearrays,float** d_sourcearrays,
//		       int* h_islice_selected_source, int** d_islice_selected_source,
//		       int* h_ispec_selected_source, int** d_ispec_selected_source,
//           int SIMULATION_TYPE){
//
//TRACE("prepare_constants");
//  
//  // EPIK_USER_REG(r_name,"compute_forces");
//  // EPIK_USER_REG(r_name,
//  
//  /* Assuming NGLLX=5. Padded is then 128 (5^3+3) */
//  int size_padded = 128*NSPEC_AB;
//  int size = NGLLX*NGLLX*NGLLX*NSPEC_AB;
//
//  // mesh  
//  print_CUDA_error_if_any(cudaMalloc((void**) d_xix, size_padded*sizeof(float)),5);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_xiy, size_padded*sizeof(float)),6);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_xiz, size_padded*sizeof(float)),7);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_etax, size_padded*sizeof(float)),8);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_etay, size_padded*sizeof(float)),9);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_etaz, size_padded*sizeof(float)),10);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_gammax, size_padded*sizeof(float)),11);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_gammay, size_padded*sizeof(float)),12);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_gammaz, size_padded*sizeof(float)),13);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_kappav, size_padded*sizeof(float)),14); 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_muv, size_padded*sizeof(float)),15);	 
//  print_CUDA_error_if_any(cudaMalloc((void**) d_ibool, size_padded*sizeof(int)),16);
//
////  print_CUDA_error_if_any(cudaMalloc((void**) d_phase_ispec_inner_elastic, num_phase_ispec_elastic*2*sizeof(int)),17);
////  print_CUDA_error_if_any(cudaMalloc((void**) d_rmass, NGLOB_AB*sizeof(float)),17);
//
//  // absorbing boundaries
//  if( num_abs_boundary_faces > 0 ){
//    print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_ispec,
//                                       num_abs_boundary_faces*sizeof(int)),769);
//    print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_ispec, h_abs_boundary_ispec,
//                                       num_abs_boundary_faces*sizeof(int),cudaMemcpyHostToDevice),770);
//    
//    print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_ijk,
//                                       3*25*num_abs_boundary_faces*sizeof(int)),772);
//    print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_ijk, h_abs_boundary_ijk,
//                                       3*25*num_abs_boundary_faces*sizeof(int),cudaMemcpyHostToDevice),773);
//    
//    print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_normal,
//                                       3*25*num_abs_boundary_faces*sizeof(int)),783);
//    print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_normal, h_abs_boundary_normal,
//                                       3*25*num_abs_boundary_faces*sizeof(int),cudaMemcpyHostToDevice),783);
//    
//    print_CUDA_error_if_any(cudaMalloc((void**) d_abs_boundary_jacobian2Dw,
//                                       25*num_abs_boundary_faces*sizeof(float)),784);
//    print_CUDA_error_if_any(cudaMemcpy(*d_abs_boundary_jacobian2Dw, h_abs_boundary_jacobian2Dw,
//                                       25*num_abs_boundary_faces*sizeof(float),cudaMemcpyHostToDevice),784);
//  }
//  
//
///*  
//  print_CUDA_error_if_any(cudaMalloc((void**) d_rho_vp, size*sizeof(float)),5);
//  print_CUDA_error_if_any(cudaMalloc((void**) d_rho_vs, size*sizeof(float)),6);
//  print_CUDA_error_if_any(cudaMemcpy(*d_rho_vp,h_rho_vp,size*sizeof(float),
//				     cudaMemcpyHostToDevice),5);
//  print_CUDA_error_if_any(cudaMemcpy(*d_rho_vs,h_rho_vs,size*sizeof(float),
//				     cudaMemcpyHostToDevice),5);
//  print_CUDA_error_if_any(cudaMalloc((void**) d_b_absorb_field, 3*25*b_num_abs_boundary_faces*sizeof(float)),7);
//  print_CUDA_error_if_any(cudaMemcpy(*d_b_absorb_field, h_b_absorb_field,
//				     3*25*b_num_abs_boundary_faces*sizeof(float),
//				     cudaMemcpyHostToDevice),7);
//  
//  print_CUDA_error_if_any(cudaMemcpy(*d_rmass,h_rmass,NGLOB_AB*sizeof(float),cudaMemcpyHostToDevice),18);
//*/
//
//  // prepare interprocess-edge exchange information
//  print_CUDA_error_if_any(cudaMalloc((void**) d_nibool_interfaces_ext_mesh,
//				     num_interfaces_ext_mesh*sizeof(int)),19);
//  print_CUDA_error_if_any(cudaMemcpy(*d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
//				     num_interfaces_ext_mesh*sizeof(int),cudaMemcpyHostToDevice),19);
//  
//  print_CUDA_error_if_any(cudaMalloc((void**) d_ibool_interfaces_ext_mesh,
//				     num_interfaces_ext_mesh*max_nibool_interfaces_ext_mesh*sizeof(int)),20);
//  print_CUDA_error_if_any(cudaMemcpy(*d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
//				     num_interfaces_ext_mesh*max_nibool_interfaces_ext_mesh*sizeof(int),
//				     cudaMemcpyHostToDevice),20);
//
//  print_CUDA_error_if_any(cudaMalloc((void**) d_ispec_is_inner,NSPEC_AB*sizeof(int)),21);
//  print_CUDA_error_if_any(cudaMemcpy(*d_ispec_is_inner, h_ispec_is_inner,
//				     NSPEC_AB*sizeof(int),
//				     cudaMemcpyHostToDevice),21);
//
///*  print_CUDA_error_if_any(cudaMalloc((void**) d_ispec_is_elastic,NSPEC_AB*sizeof(int)),21);
//  print_CUDA_error_if_any(cudaMemcpy(*d_ispec_is_elastic, h_ispec_is_elastic,
//				     NSPEC_AB*sizeof(int),
//				     cudaMemcpyHostToDevice),21);
//*/
//
//  print_CUDA_error_if_any(cudaMemcpy(*d_ibool, h_ibool,
//				     size*sizeof(int)  ,cudaMemcpyHostToDevice),512);    
//
//  // sources
//  if (SIMULATION_TYPE == 1  || SIMULATION_TYPE == 3){
//    print_CUDA_error_if_any(cudaMalloc((void**)d_sourcearrays, sizeof(float)*NSOURCES*3*125),22);
//    print_CUDA_error_if_any(cudaMemcpy(*d_sourcearrays, h_sourcearrays, sizeof(float)*NSOURCES*3*125,
//                                       cudaMemcpyHostToDevice),522);
//  }
//  
//
//  print_CUDA_error_if_any(cudaMalloc((void**)d_islice_selected_source, sizeof(int)*NSOURCES),23);
//  print_CUDA_error_if_any(cudaMemcpy(*d_islice_selected_source, h_islice_selected_source, sizeof(int)*NSOURCES,
//                                     cudaMemcpyHostToDevice),523);
//  
//  print_CUDA_error_if_any(cudaMalloc((void**)d_ispec_selected_source, sizeof(int)*NSOURCES),24);
//  print_CUDA_error_if_any(cudaMemcpy(*d_ispec_selected_source, h_ispec_selected_source,sizeof(int)*NSOURCES,
//                                     cudaMemcpyHostToDevice),524);
//  
//  // transfer constant element data with padding
//  for(int i=0;i<NSPEC_AB;i++) {
//    print_CUDA_error_if_any(cudaMemcpy(*d_xix + i*128, &h_xix[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),70);
//    print_CUDA_error_if_any(cudaMemcpy(*d_xiy+i*128,   &h_xiy[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),71);
//    print_CUDA_error_if_any(cudaMemcpy(*d_xiz+i*128,   &h_xiz[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),72);
//    print_CUDA_error_if_any(cudaMemcpy(*d_etax+i*128,  &h_etax[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),73);
//    print_CUDA_error_if_any(cudaMemcpy(*d_etay+i*128,  &h_etay[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),74);
//    print_CUDA_error_if_any(cudaMemcpy(*d_etaz+i*128,  &h_etaz[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),75);
//    print_CUDA_error_if_any(cudaMemcpy(*d_gammax+i*128,&h_gammax[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),76);
//    print_CUDA_error_if_any(cudaMemcpy(*d_gammay+i*128,&h_gammay[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),78);
//    print_CUDA_error_if_any(cudaMemcpy(*d_gammaz+i*128,&h_gammaz[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),79);
//    print_CUDA_error_if_any(cudaMemcpy(*d_kappav+i*128,&h_kappav[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),80);
//    print_CUDA_error_if_any(cudaMemcpy(*d_muv+i*128,   &h_muv[i*125],
//                                       125*sizeof(float),cudaMemcpyHostToDevice),81);      
//  }
//
////  print_CUDA_error_if_any(cudaMemcpy(*d_phase_ispec_inner_elastic, h_phase_ispec_inner_elastic, num_phase_ispec_elastic*2*sizeof(int),cudaMemcpyHostToDevice),13);
//  
//#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
//  exit_on_cuda_error("prepare_constants");
//#endif        
//}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_constants_device,
              PREPARE_CONSTANTS_DEVICE)(long* Mesh_pointer,
                                        int* h_NGLLX, 
                                        int* NSPEC_AB, int* NGLOB_AB,
                                        float* h_xix, float* h_xiy, float* h_xiz,
                                        float* h_etax, float* h_etay, float* h_etaz,
                                        float* h_gammax, float* h_gammay, float* h_gammaz,
                                        float* h_kappav, float* h_muv,
                                        int* h_ibool, 
                                        //int* h_phase_ispec_inner_elastic,int* num_phase_ispec_elastic,
                                        //float* h_rmass,
                                        int* num_interfaces_ext_mesh, int* max_nibool_interfaces_ext_mesh,
                                        int* h_nibool_interfaces_ext_mesh, int* h_ibool_interfaces_ext_mesh,
                                        float* h_hprime_xx, 
                                        float* h_hprime_yy, 
                                        float* h_hprime_zz, 
                                        float* h_hprimewgll_xx,
                                        float* h_wgllwgll_xy, 
                                        float* h_wgllwgll_xz,
                                        float* h_wgllwgll_yz,            
                                        //float* h_hprime_xx, float* h_hprimewgll_xx,
                                        //float* h_wgllwgll_xy, float* h_wgllwgll_xz,
                                        //float* h_wgllwgll_yz,
                                        int* h_abs_boundary_ispec, int* h_abs_boundary_ijk,
                                        float* h_abs_boundary_normal,
                                        //float* h_rho_vp,
                                        //float* h_rho_vs,
                                        float* h_abs_boundary_jacobian2Dw,
                                        float* h_b_absorb_field,
                                        int* num_abs_boundary_faces, int* b_num_abs_boundary_faces,
                                        int* h_ispec_is_inner, 
                                        //int* h_ispec_is_elastic,
                                        int* NSOURCES,
                                        float* h_sourcearrays,
                                        int* h_islice_selected_source,
                                        int* h_ispec_selected_source,
                                        int* h_number_receiver_global,
                                        int* h_ispec_selected_rec,
                                        int* nrec_f,
                                        int* nrec_local_f,
                                        int* SIMULATION_TYPE) {

TRACE("prepare_constants_device");
  
  int device_count,procid;
  
  // cuda initialization (needs -lcuda library)
  cuInit(0);
  
  // Gets number of GPU devices     
  cudaGetDeviceCount(&device_count);
  //printf("Cuda Devices: %d\n", device_count);
  
  // Gets rank number of MPI process
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  // Sets the active device 
  if(device_count > 1) {
    // daniel: todo - generalize for more GPUs per node?
    // assumes we have 2 GPU devices per node and running 2 MPI processes per node as well
    cudaSetDevice((procid)%2);
    exit_on_cuda_error("cudaSetDevice");   
  }

  //printf("GPU_MODE Active. Preparing Fields and Constants on Device.\n");

  // allocates mesh parameter structure  
  Mesh* mp = (Mesh*)malloc(sizeof(Mesh));
  *Mesh_pointer = (long)mp;

  // checks if NGLLX == 5
  if( *h_NGLLX != NGLLX ){
    exit_on_cuda_error("NGLLX must be 5 for CUDA devices");   
  }
  
  // sets global parameters  
  mp->NSPEC_AB = *NSPEC_AB;
  mp->NGLOB_AB = *NGLOB_AB;
  
  //mp->d_num_phase_ispec_elastic = *num_phase_ispec_elastic;

  // sets constant arrays
  setConst_hprime_xx(h_hprime_xx,mp);
  setConst_hprime_yy(h_hprime_yy,mp);
  setConst_hprime_zz(h_hprime_zz,mp);
  setConst_hprimewgll_xx(h_hprimewgll_xx,mp);
  setConst_wgllwgll_xy(h_wgllwgll_xy,mp);
  setConst_wgllwgll_xz(h_wgllwgll_xz,mp);
  setConst_wgllwgll_yz(h_wgllwgll_yz,mp);
  
/*  setConst_hprime_xx    (h_hprime_xx    );
  setConst_hprimewgll_xx(h_hprimewgll_xx);
  setConst_wgllwgll_xy  (h_wgllwgll_xy,mp);
  setConst_wgllwgll_xz  (h_wgllwgll_xz,mp);
  setConst_wgllwgll_yz  (h_wgllwgll_yz,mp);
*/
  
  /* Assuming NGLLX=5. Padded is then 128 (5^3+3) */
  int size_padded = 128 * (*NSPEC_AB);
  int size = 125 * (*NSPEC_AB);

  // mesh    
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

    
//  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_phase_ispec_inner_elastic, *num_phase_ispec_elastic*2*sizeof(int)),17);
//  print_CUDA_error_if_any(cudaMemcpy(mp->d_phase_ispec_inner_elastic, h_phase_ispec_inner_elastic, *num_phase_ispec_elastic*2*sizeof(int),cudaMemcpyHostToDevice),13);
  
//  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_rmass, *NGLOB_AB*sizeof(float)),17);
//  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass,h_rmass,*NGLOB_AB*sizeof(float),cudaMemcpyHostToDevice),18);

  // absorbing boundaries
  if( *num_abs_boundary_faces > 0 ){  
    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_ispec,
               (*num_abs_boundary_faces)*sizeof(int)),771);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ispec, h_abs_boundary_ispec,
               (*num_abs_boundary_faces)*sizeof(int),
               cudaMemcpyHostToDevice),771);
    
    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_ijk,
               3*25*(*num_abs_boundary_faces)*sizeof(int)),772);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ijk, h_abs_boundary_ijk,
               3*25*(*num_abs_boundary_faces)*sizeof(int),
               cudaMemcpyHostToDevice),772);
    
    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_normal,
               3*25*(*num_abs_boundary_faces)*sizeof(int)),773);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_normal, h_abs_boundary_normal,
               3*25*(*num_abs_boundary_faces)*sizeof(int),
               cudaMemcpyHostToDevice),773);
    
    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_abs_boundary_jacobian2Dw,
               25*(*num_abs_boundary_faces)*sizeof(float)),774);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_jacobian2Dw, h_abs_boundary_jacobian2Dw,
               25*(*num_abs_boundary_faces)*sizeof(float),
               cudaMemcpyHostToDevice),774);  
  }

/*  
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
*/  

  // prepare interprocess-edge exchange information
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_nibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh*sizeof(int)),19);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh*sizeof(int),cudaMemcpyHostToDevice),19);

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool_interfaces_ext_mesh,
                                     *num_interfaces_ext_mesh* *max_nibool_interfaces_ext_mesh*
                                     sizeof(int)),20);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
				     *num_interfaces_ext_mesh* *max_nibool_interfaces_ext_mesh*sizeof(int),
				     cudaMemcpyHostToDevice),20);

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ispec_is_inner,*NSPEC_AB*sizeof(int)),21);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_inner, h_ispec_is_inner,
				     *NSPEC_AB*sizeof(int),
				     cudaMemcpyHostToDevice),21);
             
//  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ispec_is_elastic,*NSPEC_AB*sizeof(int)),21);
//  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_elastic, h_ispec_is_elastic,
//				     *NSPEC_AB*sizeof(int),
//				     cudaMemcpyHostToDevice),21);

  print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool, h_ibool,
				     size*sizeof(int)  ,cudaMemcpyHostToDevice),22);    

  // sources
  if (*SIMULATION_TYPE == 1  || *SIMULATION_TYPE == 3){
    // not needed in case of pure adjoint simulations (SIMULATION_TYPE == 2)
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_sourcearrays, sizeof(float)* *NSOURCES*3*125),522);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_sourcearrays, h_sourcearrays, sizeof(float)* *NSOURCES*3*125,
                                       cudaMemcpyHostToDevice),522);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_stf_pre_compute),
                                       *NSOURCES*sizeof(double)),525);
  }

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_source, sizeof(int) * *NSOURCES),523);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_islice_selected_source, h_islice_selected_source, sizeof(int)* *NSOURCES,
				     cudaMemcpyHostToDevice),523);

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_ispec_selected_source, sizeof(int)* *NSOURCES),524);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_selected_source, h_ispec_selected_source,sizeof(int)* *NSOURCES,
				     cudaMemcpyHostToDevice),524);

  
  // transfer constant element data with padding
  for(int i=0;i<*NSPEC_AB;i++) {
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xix + i*128, &h_xix[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),70);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiy+i*128,   &h_xiy[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),71);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiz+i*128,   &h_xiz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),72);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etax+i*128,  &h_etax[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),73);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etay+i*128,  &h_etay[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),74);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etaz+i*128,  &h_etaz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),75);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammax+i*128,&h_gammax[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),76);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammay+i*128,&h_gammay[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),77);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammaz+i*128,&h_gammaz[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),78);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_kappav+i*128,&h_kappav[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),79);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_muv+i*128,   &h_muv[i*125],
				       125*sizeof(float),cudaMemcpyHostToDevice),80);
      
  }
        

  // receiver stations
  int nrec = *nrec_f; // total number of receivers
  int nrec_local = *nrec_local_f;	// number of receiver located in this partition
  // note that:
  // size(number_receiver_global) = nrec_local
  // size(ispec_selected_rec) = nrec  
  mp->nrec_local = nrec_local;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_number_receiver_global),nrec_local*sizeof(int)),1);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_number_receiver_global,h_number_receiver_global,
                                     nrec_local*sizeof(int),cudaMemcpyHostToDevice),602);  
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_selected_rec),nrec*sizeof(int)),603);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_selected_rec,h_ispec_selected_rec,
                                     nrec*sizeof(int),cudaMemcpyHostToDevice),604);  

//  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_station_seismo_field),3*125*nrec_local*sizeof(float)),605);
//  mp->h_station_seismo_field = (float*)malloc(3*125*nrec_local*sizeof(float));

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_constants_device");
#endif            
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_and_transfer_noise_backward_fields,
              PREPARE_AND_TRANSFER_NOISE_BACKWARD_FIELDS)(long* Mesh_pointer_f,
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

TRACE("prepare_and_transfer_noise_backward_fields_");
                  
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

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_and_transfer_noise_backward_constants,
              PREPARE_AND_TRANSFER_NOISE_BACKWARD_CONSTANTS)(long* Mesh_pointer_f,
                                                            float* normal_x_noise,
                                                            float* normal_y_noise,
                                                            float* normal_z_noise,
                                                            float* mask_noise,
                                                            float* free_surface_jacobian2Dw,
                                                            int* nfaces_surface_ext_mesh
                                                            ) {

TRACE("prepare_and_transfer_noise_backward_constants_");

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
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  printf("jacobian_size = %d\n",25*(*nfaces_surface_ext_mesh));
  exit_on_cuda_error("prepare_and_transfer_noise_backward_constants_");  
#endif  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_noise_constants_device,
              PREPARE_NOISE_CONSTANTS_DEVICE)(long* Mesh_pointer_f, 
                                              int* h_NGLLX,
                                              int* NSPEC_AB, int* NGLOB_AB,
                                              int* free_surface_ispec,int* free_surface_ijk,
                                              int* num_free_surface_faces,
                                              int* size_free_surface_ijk, int* SIMULATION_TYPE) {

TRACE("prepare_noise_constants_device_");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  mp->num_free_surface_faces = *num_free_surface_faces;

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ispec, *num_free_surface_faces*sizeof(int)),1);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec, free_surface_ispec, *num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),1);

  // alloc storage for the surface buffer to be copied
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_noise_surface_movie, 3*25*(*num_free_surface_faces)*sizeof(float)),1);

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ijk, (*size_free_surface_ijk)*sizeof(float)),1);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,(*size_free_surface_ijk)*sizeof(float),cudaMemcpyHostToDevice),1);
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_sensitivity_kernels,
              PREPARE_SENSITIVITY_KERNELS)(long* Mesh_pointer_f,
                                           float* rho_kl,
                                           float* mu_kl,
                                           float* kappa_kl,
                                           float* epsilon_trace_over_3,
                                           float* b_epsilon_trace_over_3,
                                           float* Sigma_kl,
                                           int* NSPEC_ADJOINTf) {

TRACE("prepare_sensitivity_kernels_");
               
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  int NSPEC_ADJOINT = *NSPEC_ADJOINTf;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_kl),
				     125*mp->NSPEC_AB*sizeof(float)),800);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_mu_kl),
				     125*mp->NSPEC_AB*sizeof(float)),801);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_kl),
				     125*mp->NSPEC_AB*sizeof(float)),802);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_epsilon_trace_over_3),
				     125*mp->NSPEC_AB*sizeof(float)),803);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilon_trace_over_3),
				     125*mp->NSPEC_AB*sizeof(float)),804);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_Sigma_kl),
				     125*(mp->NSPEC_AB)*sizeof(float)),805);

  cudaMemcpy(mp->d_rho_kl,rho_kl, 125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_mu_kl,mu_kl, 125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_kappa_kl,kappa_kl, 125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_epsilon_trace_over_3,epsilon_trace_over_3,
	     125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_b_epsilon_trace_over_3 ,b_epsilon_trace_over_3,
	     125*NSPEC_ADJOINT*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(mp->d_Sigma_kl, Sigma_kl, 125*(NSPEC_ADJOINT)*sizeof(float),
	     cudaMemcpyHostToDevice);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_sensitivity_kernels");
#endif
}
					     
/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_adjoint_constants_device,
              PREPARE_ADJOINT_CONSTANTS_DEVICE)(long* Mesh_pointer_f,
                                                //int* ispec_selected_rec,
                                                //int* islice_selected_rec,
                                                //int* islice_selected_rec_size,
                                                //int* nrec,
                                                float* noise_sourcearray,
                                                int* NSTEP,
                                                float* epsilondev_xx,
                                                float* epsilondev_yy,
                                                float* epsilondev_xy,
                                                float* epsilondev_xz,
                                                float* epsilondev_yz,
                                                int* NSPEC_STRAIN_ONLY
                                                ) {
TRACE("prepare_adjoint_constants_device_");
              
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_adjoint_constants_device 1");  
#endif

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  int epsilondev_size = 128*(*NSPEC_STRAIN_ONLY);
  
  // already done earlier
  // print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_ispec_selected_rec,
  // *nrec*sizeof(int)),1);
  // cudaMemcpy(mp->d_ispec_selected_rec,ispec_selected_rec, *nrec*sizeof(int),  
  // cudaMemcpyHostToDevice);

  //print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_rec,
	//			     *islice_selected_rec_size*sizeof(int)),901);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_noise_sourcearray,
				     3*125*(*NSTEP)*sizeof(float)),902);

  
  cudaMemcpy(mp->d_noise_sourcearray, noise_sourcearray,
	     3*125*(*NSTEP)*sizeof(float),
	     cudaMemcpyHostToDevice);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_adjoint_constants_device 2");  
#endif
  
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xx,
				     epsilondev_size*sizeof(float)),903);
  cudaMemcpy(mp->d_epsilondev_xx,epsilondev_xx,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yy,
				     epsilondev_size*sizeof(float)),904);
  cudaMemcpy(mp->d_epsilondev_yy,epsilondev_yy,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xy,
				     epsilondev_size*sizeof(float)),905);
  cudaMemcpy(mp->d_epsilondev_xy,epsilondev_xy,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xz,
				     epsilondev_size*sizeof(float)),906);
  cudaMemcpy(mp->d_epsilondev_xz,epsilondev_xz,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yz,
				     epsilondev_size*sizeof(float)),907);
  cudaMemcpy(mp->d_epsilondev_yz,epsilondev_yz,epsilondev_size*sizeof(float),
	     cudaMemcpyHostToDevice);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING       
  exit_on_cuda_error("prepare_adjoint_constants_device 3");  
#endif  
  
  // these don't seem necessary and crash code for NOISE_TOMOGRAPHY >
  // 0 b/c rho_kl, etc not yet allocated when NT=1
}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(prepare_adjoint_sim2_or_3_constants_device,
              PREPARE_ADJOINT_SIM2_OR_3_CONSTANTS_DEVICE)(
                                                          long* Mesh_pointer_f,
                                                          int* islice_selected_rec,
                                                          int* islice_selected_rec_size) {
  
TRACE("prepare_adjoint_sim2_or_3_constants_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  
  // allocates arrays for receivers
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_rec,
                                     *islice_selected_rec_size*sizeof(int)),802);
  
  // copies arrays to GPU device
  print_CUDA_error_if_any(cudaMemcpy(mp->d_islice_selected_rec,islice_selected_rec, 
                                     *islice_selected_rec_size*sizeof(int),cudaMemcpyHostToDevice),804);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING       
  exit_on_cuda_error("prepare_adjoint_sim2_or_3_constants_device");  
#endif  
}  

/* ----------------------------------------------------------------------------------------------- */

/*
extern "C" {
  void prepare_fields_device_(long* Mesh_pointer_f, int* size);
  void transfer_fields_to_device_(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f);
  void transfer_fields_from_device_(int* size, float* displ, float* veloc, float* accel,long* Mesh_pointer_f);
}
*/

/* ----------------------------------------------------------------------------------------------- */

/*
void prepare_fields_device_(long* Mesh_pointer_f, int* size) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_displ),sizeof(float)*(*size)),0);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_veloc),sizeof(float)*(*size)),1);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_accel),sizeof(float)*(*size)),2);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_accel_buffer),sizeof(float)*(*size)),2);

}
*/

/* ----------------------------------------------------------------------------------------------- */

// for ELASTIC simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
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
                                             int* b_num_abs_boundary_faces){
  
TRACE("prepare_fields_elastic_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  /* Assuming NGLLX==5. Padded is then 128 (5^3+3) */  
  int size_padded = 128 * mp->NSPEC_AB;
  int size_nonpadded = 125 * mp->NSPEC_AB;
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_displ),sizeof(float)*(*size)),200);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_veloc),sizeof(float)*(*size)),201);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_accel),sizeof(float)*(*size)),202);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_accel_buffer),sizeof(float)*(*size)),203);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass),sizeof(float)*mp->NGLOB_AB),204);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_vp),size_padded*sizeof(float)),205); 
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_vs),size_padded*sizeof(float)),206); 
  
  mp->d_num_phase_ispec_elastic = *num_phase_ispec_elastic;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_phase_ispec_inner_elastic), mp->d_num_phase_ispec_elastic*2*sizeof(int)),207);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_is_elastic),mp->NSPEC_AB*sizeof(int)),208);
  
  // transfer element data
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass,rmass,
                                     sizeof(float)*mp->NGLOB_AB,cudaMemcpyHostToDevice),209);  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_phase_ispec_inner_elastic,phase_ispec_inner_elastic, 
                                     mp->d_num_phase_ispec_elastic*2*sizeof(int),cudaMemcpyHostToDevice),210);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_elastic,ispec_is_elastic,
                                     mp->NSPEC_AB*sizeof(int),cudaMemcpyHostToDevice),211);
  
  // daniel: not sure if rho_vp, rho_vs needs padding... they are needed for stacey boundary condition
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vp, rho_vp,
                                     size_nonpadded*sizeof(float),cudaMemcpyHostToDevice),212);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vs, rho_vs,
                                     size_nonpadded*sizeof(float),cudaMemcpyHostToDevice),213);
  
  // for seismograms
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_station_seismo_field),3*125*(mp->nrec_local)*sizeof(float)),214);
  mp->h_station_seismo_field = (float*)malloc(3*125*(mp->nrec_local)*sizeof(float));
  
  // absorbing conditions
  if( *ABSORBING_CONDITIONS == 1 ){
    mp->b_num_abs_boundary_faces = *b_num_abs_boundary_faces;
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_absorb_field), 
                                       3*25*mp->b_num_abs_boundary_faces*sizeof(float)),791);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_absorb_field, h_b_absorb_field,
                                       3*25*mp->b_num_abs_boundary_faces*sizeof(float),cudaMemcpyHostToDevice),792);
  }
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING       
  exit_on_cuda_error("prepare_fields_elastic_device");  
#endif    
}


/* ----------------------------------------------------------------------------------------------- */

// for ACOUSTIC simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_fields_acoustic_device,
              PREPARE_FIELDS_ACOUSTIC_DEVICE)(long* Mesh_pointer_f, 
                                               float* rmass_acoustic, 
                                               float* rhostore,
                                               float* kappastore,
                                               int* num_phase_ispec_acoustic, 
                                               int* phase_ispec_inner_acoustic,
                                               int* ispec_is_acoustic,
                                               int* num_free_surface_faces,
                                               int* free_surface_ispec,
                                               int* free_surface_ijk,
                                               int* ABSORBING_CONDITIONS,
                                               int* b_reclen_potential,
                                               float* b_absorb_potential,
                                               int* SIMULATION_TYPE,
                                               float* rho_ac_kl,
                                               float* kappa_ac_kl) {
  
TRACE("prepare_fields_acoustic_device");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  /* Assuming NGLLX==5. Padded is then 128 (5^3+3) */
  int size_padded = 128 * mp->NSPEC_AB;
  int size_nonpadded = 125 * mp->NSPEC_AB;
  int size = mp->NGLOB_AB;
  
  // allocates arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_acoustic),sizeof(float)*size),100);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_dot_acoustic),sizeof(float)*size),101);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_dot_dot_acoustic),sizeof(float)*size),102);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_potential_dot_dot_buffer),sizeof(float)*size),103);
  
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass_acoustic),sizeof(float)*size),104);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rhostore),size_padded*sizeof(float)),105); 
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappastore),size_padded*sizeof(float)),106); 
  
  mp->num_phase_ispec_acoustic = *num_phase_ispec_acoustic;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_phase_ispec_inner_acoustic), mp->num_phase_ispec_acoustic*2*sizeof(int)),107);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_is_acoustic),mp->NSPEC_AB*sizeof(int)),108);
  
  mp->num_free_surface_faces = *num_free_surface_faces;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ispec),mp->num_free_surface_faces*sizeof(int)),109);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ijk),3*25*mp->num_free_surface_faces*sizeof(int)),110);
  
  // absorbing boundaries
  if( *ABSORBING_CONDITIONS == 1 ){
    mp->d_b_reclen_potential = *b_reclen_potential;
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_absorb_potential),mp->d_b_reclen_potential),111); 
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_absorb_potential,b_absorb_potential,
                                       mp->d_b_reclen_potential,cudaMemcpyHostToDevice),112);    
  }
  
  // kernel simulations
  if( *SIMULATION_TYPE == 3 ){
    // allocates backward/reconstructed arrays on device (GPU)
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_acoustic),sizeof(float)*size),113);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_dot_acoustic),sizeof(float)*size),114);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_dot_dot_acoustic),sizeof(float)*size),115);    
    
    // allocates kernels  
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_ac_kl),125*mp->NSPEC_AB*sizeof(float)),181);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_ac_kl),125*mp->NSPEC_AB*sizeof(float)),182);
    // copies over initial values
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_ac_kl,rho_ac_kl, 
                                       125*mp->NSPEC_AB*sizeof(float),cudaMemcpyHostToDevice),183);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_kappa_ac_kl,kappa_ac_kl, 
                                       125*mp->NSPEC_AB*sizeof(float),cudaMemcpyHostToDevice),184);
    
  }
  
  // transfer element data
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass_acoustic,rmass_acoustic,
                                     sizeof(float)*size,cudaMemcpyHostToDevice),116);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_phase_ispec_inner_acoustic,phase_ispec_inner_acoustic, 
                                     mp->num_phase_ispec_acoustic*2*sizeof(int),cudaMemcpyHostToDevice),117);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_acoustic,ispec_is_acoustic,
                                     mp->NSPEC_AB*sizeof(int),cudaMemcpyHostToDevice),118);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec,free_surface_ispec,
                                     mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),119);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                     3*25*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),120);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_kappastore,kappastore,
                                     size_nonpadded*sizeof(float),cudaMemcpyHostToDevice),121);
  
  // transfer constant element data with padding
  for(int i=0;i<mp->NSPEC_AB;i++) {  
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rhostore+i*128, &rhostore[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),122);
  }
  
  // for seismograms
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_station_seismo_potential),mp->nrec_local*125*sizeof(float)),123);
  mp->h_station_seismo_potential = (float*)malloc(mp->nrec_local*125*sizeof(float));
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING       
  exit_on_cuda_error("prepare_fields_acoustic_device");  
#endif    
}

