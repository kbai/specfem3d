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

void exit_on_error(char* info)
{
  printf("\nERROR: %s\n",info);
  fflush(stdout);
#ifdef USE_MPI
  MPI_Abort(MPI_COMM_WORLD,1);
#endif
  exit(EXIT_FAILURE);
  return;
}

/* ----------------------------------------------------------------------------------------------- */

void print_CUDA_error_if_any(cudaError_t err, int num)
{
  if (cudaSuccess != err)
  {
    printf("\nCUDA error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",cudaGetErrorString(err),num);
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
              get_FREE_DEVICE_MEMORY)(realw* free, realw* used, realw* total ) {
TRACE("get_free_device_memory");

  double free_db,used_db,total_db;

  get_free_memory(&free_db,&used_db,&total_db);

  // converts to MB
  *free = (realw) free_db/1024.0/1024.0;
  *used = (realw) used_db/1024.0/1024.0;
  *total = (realw) total_db/1024.0/1024.0;
  return;
}


/* ----------------------------------------------------------------------------------------------- */
//daniel: helper function
/*
__global__ void check_phase_ispec_kernel(int num_phase_ispec,
                                         int* phase_ispec,
                                         int NSPEC_AB,
                                         int* ier) {

  int i,ispec,iphase,count0,count1;
  *ier = 0;

  for(iphase=0; iphase < 2; iphase++){
    count0 = 0;
    count1 = 0;

    for(i=0; i < num_phase_ispec; i++){
      ispec = phase_ispec[iphase*num_phase_ispec + i] - 1;
      if( ispec < -1 || ispec >= NSPEC_AB ){
        printf("Error in d_phase_ispec_inner_elastic %d %d\n",i,ispec);
        *ier = 1;
        return;
      }
      if( ispec >= 0 ){ count0++;}
      if( ispec < 0 ){ count1++;}
    }

    printf("check_phase_ispec done: phase %d, count = %d %d \n",iphase,count0,count1);

  }
}

void check_phase_ispec(long* Mesh_pointer_f,int type){

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container

  printf("check phase_ispec for type=%d\n",type);

  dim3 grid(1,1);
  dim3 threads(1,1,1);

  int* h_debug = (int*) calloc(1,sizeof(int));
  int* d_debug;
  cudaMalloc((void**)&d_debug,sizeof(int));

  if( type == 1 ){
    check_phase_ispec_kernel<<<grid,threads>>>(mp->num_phase_ispec_elastic,
                                             mp->d_phase_ispec_inner_elastic,
                                             mp->NSPEC_AB,
                                             d_debug);
  }else if( type == 2 ){
    check_phase_ispec_kernel<<<grid,threads>>>(mp->num_phase_ispec_acoustic,
                                               mp->d_phase_ispec_inner_acoustic,
                                               mp->NSPEC_AB,
                                               d_debug);
  }

  cudaMemcpy(h_debug,d_debug,1*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(d_debug);
  if( *h_debug != 0 ){printf("error for type=%d\n",type); exit(1);}
  free(h_debug);
  fflush(stdout);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("check_phase_ispec");
#endif

}
*/

/* ----------------------------------------------------------------------------------------------- */
//daniel: helper function
/*
__global__ void check_ispec_is_kernel(int NSPEC_AB,
                                      int* ispec_is,
                                      int* ier) {

  int ispec,count0,count1;

  *ier = 0;
  count0 = 0;
  count1 = 0;
  for(ispec=0; ispec < NSPEC_AB; ispec++){
    if( ispec_is[ispec] < -1 || ispec_is[ispec] > 1 ){
      printf("Error in ispec_is %d %d\n",ispec,ispec_is[ispec]);
      *ier = 1;
      return;
      //exit(1);
    }
    if( ispec_is[ispec] == 0 ){count0++;}
    if( ispec_is[ispec] != 0 ){count1++;}
  }
  printf("check_ispec_is done: count = %d %d\n",count0,count1);
}

void check_ispec_is(long* Mesh_pointer_f,int type){

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container

  printf("check ispec_is for type=%d\n",type);

  dim3 grid(1,1);
  dim3 threads(1,1,1);

  int* h_debug = (int*) calloc(1,sizeof(int));
  int* d_debug;
  cudaMalloc((void**)&d_debug,sizeof(int));

  if( type == 0 ){
    check_ispec_is_kernel<<<grid,threads>>>(mp->NSPEC_AB,
                                            mp->d_ispec_is_inner,
                                            d_debug);
  }else if( type == 1 ){
    check_ispec_is_kernel<<<grid,threads>>>(mp->NSPEC_AB,
                                            mp->d_ispec_is_elastic,
                                            d_debug);
  }else if( type == 2 ){
    check_ispec_is_kernel<<<grid,threads>>>(mp->NSPEC_AB,
                                            mp->d_ispec_is_acoustic,
                                            d_debug);
  }

  cudaMemcpy(h_debug,d_debug,1*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(d_debug);
  if( *h_debug != 0 ){printf("error for type=%d\n",type); exit(1);}
  free(h_debug);
  fflush(stdout);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("check_ispec_is");
#endif
}
*/
/* ----------------------------------------------------------------------------------------------- */
//daniel: helper function
/*
__global__ void check_array_ispec_kernel(int num_array_ispec,
                                         int* array_ispec,
                                         int NSPEC_AB,
                                         int* ier) {

  int i,ispec,count0,count1;

  *ier = 0;
  count0 = 0;
  count1 = 0;

  for(i=0; i < num_array_ispec; i++){
    ispec = array_ispec[i] - 1;
    if( ispec < -1 || ispec >= NSPEC_AB ){
      printf("Error in d_array_ispec %d %d\n",i,ispec);
      *ier = 1;
      return;
    }
    if( ispec >= 0 ){ count0++;}
    if( ispec < 0 ){ count1++;}
  }

  printf("check_array_ispec done: count = %d %d \n",count0,count1);
}

void check_array_ispec(long* Mesh_pointer_f,int type){

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container

  printf("check array_ispec for type=%d\n",type);

  dim3 grid(1,1);
  dim3 threads(1,1,1);

  int* h_debug = (int*) calloc(1,sizeof(int));
  int* d_debug;
  cudaMalloc((void**)&d_debug,sizeof(int));

  if( type == 1 ){
    check_array_ispec_kernel<<<grid,threads>>>(mp->d_num_abs_boundary_faces,
                                               mp->d_abs_boundary_ispec,
                                               mp->NSPEC_AB,
                                               d_debug);
  }

  cudaMemcpy(h_debug,d_debug,1*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(d_debug);
  if( *h_debug != 0 ){printf("error for type=%d\n",type); exit(1);}
  free(h_debug);
  fflush(stdout);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("check_array_ispec");
#endif

}
*/

/* ----------------------------------------------------------------------------------------------- */

// GPU preparation

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_constants_device,
              PREPARE_CONSTANTS_DEVICE)(long* Mesh_pointer,
                                        int* h_NGLLX,
                                        int* NSPEC_AB, int* NGLOB_AB,
                                        realw* h_xix, realw* h_xiy, realw* h_xiz,
                                        realw* h_etax, realw* h_etay, realw* h_etaz,
                                        realw* h_gammax, realw* h_gammay, realw* h_gammaz,
                                        realw* h_kappav, realw* h_muv,
                                        int* h_ibool,
                                        int* num_interfaces_ext_mesh,
                                        int* max_nibool_interfaces_ext_mesh,
                                        int* h_nibool_interfaces_ext_mesh,
                                        int* h_ibool_interfaces_ext_mesh,
                                        realw* h_hprime_xx,realw* h_hprime_yy,realw* h_hprime_zz,
                                        realw* h_hprimewgll_xx,realw* h_hprimewgll_yy,realw* h_hprimewgll_zz,
                                        realw* h_wgllwgll_xy,realw* h_wgllwgll_xz,realw* h_wgllwgll_yz,
                                        int* ABSORBING_CONDITIONS,
                                        int* h_abs_boundary_ispec, int* h_abs_boundary_ijk,
                                        realw* h_abs_boundary_normal,
                                        realw* h_abs_boundary_jacobian2Dw,
                                        int* h_num_abs_boundary_faces,
                                        int* h_ispec_is_inner,
                                        int* NSOURCES,
                                        int* nsources_local_f,
                                        realw* h_sourcearrays,
                                        int* h_islice_selected_source,
                                        int* h_ispec_selected_source,
                                        int* h_number_receiver_global,
                                        int* h_ispec_selected_rec,
                                        int* nrec_f,
                                        int* nrec_local_f,
                                        int* SIMULATION_TYPE,
                                        int* USE_MESH_COLORING_GPU_f,
                                        int* nspec_acoustic,int* nspec_elastic,
                                        int* myrank_f,int* ncuda_devices) {

TRACE("prepare_constants_device");

  int device_count = 0;

  // cuda initialization (needs -lcuda library)
  //cuInit(0);
  CUresult status = cuInit(0);
  if ( CUDA_SUCCESS != status ) exit_on_error("CUDA device initialization failed");

  // Gets number of GPU devices
  cudaGetDeviceCount(&device_count);
  //printf("Cuda Devices: %d\n", device_count);
  if (device_count == 0) exit_on_error("There is no device supporting CUDA\n");
  *ncuda_devices = device_count;

  // Gets rank number of MPI process
  int myrank = *myrank_f;

  // Sets the active device
  if(device_count > 1) {
    // generalized for more GPUs per node
    cudaSetDevice( myrank%device_count );
    exit_on_cuda_error("cudaSetDevice");
  }

  // allocates mesh parameter structure
  Mesh* mp = (Mesh*) malloc( sizeof(Mesh) );
  if (mp == NULL) exit_on_error("error allocating mesh pointer");
  *Mesh_pointer = (long)mp;

  // checks if NGLLX == 5
  if( *h_NGLLX != NGLLX ){
    exit_on_error("NGLLX must be 5 for CUDA devices");
  }

  // sets global parameters
  mp->NSPEC_AB = *NSPEC_AB;
  mp->NGLOB_AB = *NGLOB_AB;

  // sets constant arrays
  setConst_hprime_xx(h_hprime_xx,mp);
  setConst_hprime_yy(h_hprime_yy,mp);
  setConst_hprime_zz(h_hprime_zz,mp);
  setConst_hprimewgll_xx(h_hprimewgll_xx,mp);
  setConst_hprimewgll_yy(h_hprimewgll_yy,mp);
  setConst_hprimewgll_zz(h_hprimewgll_zz,mp);
  setConst_wgllwgll_xy(h_wgllwgll_xy,mp);
  setConst_wgllwgll_xz(h_wgllwgll_xz,mp);
  setConst_wgllwgll_yz(h_wgllwgll_yz,mp);

  /* Assuming NGLLX=5. Padded is then 128 (5^3+3) */
  int size_padded = NGLL3_PADDED * (mp->NSPEC_AB);
  //int size_nonpadded = NGLL3 * (mp->NSPEC_AB);

  // mesh
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xix, size_padded*sizeof(realw)),1001);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xiy, size_padded*sizeof(realw)),1002);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xiz, size_padded*sizeof(realw)),1003);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etax, size_padded*sizeof(realw)),1004);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etay, size_padded*sizeof(realw)),1005);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etaz, size_padded*sizeof(realw)),1006);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammax, size_padded*sizeof(realw)),1007);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammay, size_padded*sizeof(realw)),1008);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammaz, size_padded*sizeof(realw)),1009);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_kappav, size_padded*sizeof(realw)),1010);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_muv, size_padded*sizeof(realw)),1011);

  // transfer constant element data with padding
  for(int i=0;i < mp->NSPEC_AB;i++) {
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xix + i*NGLL3_PADDED, &h_xix[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1501);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiy+i*NGLL3_PADDED,   &h_xiy[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1502);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiz+i*NGLL3_PADDED,   &h_xiz[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1503);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etax+i*NGLL3_PADDED,  &h_etax[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1504);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etay+i*NGLL3_PADDED,  &h_etay[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1505);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etaz+i*NGLL3_PADDED,  &h_etaz[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1506);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammax+i*NGLL3_PADDED,&h_gammax[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1507);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammay+i*NGLL3_PADDED,&h_gammay[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1508);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammaz+i*NGLL3_PADDED,&h_gammaz[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1509);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_kappav+i*NGLL3_PADDED,&h_kappav[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1510);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_muv+i*NGLL3_PADDED,   &h_muv[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),1511);
  }

  // global indexing
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool,size_padded*sizeof(int)),1021);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool, h_ibool,
                                     NGLL3*(mp->NSPEC_AB)*sizeof(int),cudaMemcpyHostToDevice),1022);


  // prepare interprocess-edge exchange information
  mp->num_interfaces_ext_mesh = *num_interfaces_ext_mesh;
  mp->max_nibool_interfaces_ext_mesh = *max_nibool_interfaces_ext_mesh;
  if( mp->num_interfaces_ext_mesh > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_nibool_interfaces_ext_mesh,
                                       (mp->num_interfaces_ext_mesh)*sizeof(int)),1201);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
                                       (mp->num_interfaces_ext_mesh)*sizeof(int),cudaMemcpyHostToDevice),1202);

    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool_interfaces_ext_mesh,
                                       (mp->num_interfaces_ext_mesh)*(mp->max_nibool_interfaces_ext_mesh)*sizeof(int)),1203);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
                                       (mp->num_interfaces_ext_mesh)*(mp->max_nibool_interfaces_ext_mesh)*sizeof(int),
                                       cudaMemcpyHostToDevice),1204);
  }

  // inner elements
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ispec_is_inner,mp->NSPEC_AB*sizeof(int)),1205);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_inner, h_ispec_is_inner,
                                     mp->NSPEC_AB*sizeof(int),cudaMemcpyHostToDevice),1206);

  // daniel: check
  //check_ispec_is(Mesh_pointer,0);

  // absorbing boundaries
  mp->d_num_abs_boundary_faces = *h_num_abs_boundary_faces;
  if( *ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**) &(mp->d_abs_boundary_ispec),
                                       (mp->d_num_abs_boundary_faces)*sizeof(int)),1101);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ispec, h_abs_boundary_ispec,
                                       (mp->d_num_abs_boundary_faces)*sizeof(int),
                                       cudaMemcpyHostToDevice),1102);

    // daniel: check
    //check_array_ispec(Mesh_pointer,1);


    print_CUDA_error_if_any(cudaMalloc((void**) &(mp->d_abs_boundary_ijk),
                                       3*NGLL2*(mp->d_num_abs_boundary_faces)*sizeof(int)),1103);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ijk, h_abs_boundary_ijk,
                                       3*NGLL2*(mp->d_num_abs_boundary_faces)*sizeof(int),
                                       cudaMemcpyHostToDevice),1104);

    print_CUDA_error_if_any(cudaMalloc((void**) &(mp->d_abs_boundary_normal),
                                       3*NGLL2*(mp->d_num_abs_boundary_faces)*sizeof(realw)),1105);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_normal, h_abs_boundary_normal,
                                       3*NGLL2*(mp->d_num_abs_boundary_faces)*sizeof(realw),
                                       cudaMemcpyHostToDevice),1106);

    print_CUDA_error_if_any(cudaMalloc((void**) &(mp->d_abs_boundary_jacobian2Dw),
                                       NGLL2*(mp->d_num_abs_boundary_faces)*sizeof(realw)),1107);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_jacobian2Dw, h_abs_boundary_jacobian2Dw,
                                       NGLL2*(mp->d_num_abs_boundary_faces)*sizeof(realw),
                                       cudaMemcpyHostToDevice),1108);
  }

  // sources
  mp->nsources_local = *nsources_local_f;
  if (*SIMULATION_TYPE == 1  || *SIMULATION_TYPE == 3){
    // not needed in case of pure adjoint simulations (SIMULATION_TYPE == 2)
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_sourcearrays,
                                       sizeof(realw)* *NSOURCES*3*NGLL3),1301);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_sourcearrays, h_sourcearrays,
                                       sizeof(realw)* *NSOURCES*3*NGLL3,cudaMemcpyHostToDevice),1302);

    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_stf_pre_compute,
                                       *NSOURCES*sizeof(double)),1303);
  }

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_source,
                                     sizeof(int) * *NSOURCES),1401);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_islice_selected_source, h_islice_selected_source,
                                     sizeof(int)* *NSOURCES,cudaMemcpyHostToDevice),1402);

  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_ispec_selected_source,
                                     sizeof(int)* *NSOURCES),1403);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_selected_source, h_ispec_selected_source,
                                     sizeof(int)* *NSOURCES,cudaMemcpyHostToDevice),1404);


  // receiver stations
  int nrec = *nrec_f; // total number of receivers
  mp->nrec_local = *nrec_local_f; // number of receiver located in this partition
  //int nrec_local = *nrec_local_f;
  // note that:
  // size(number_receiver_global) = nrec_local
  // size(ispec_selected_rec) = nrec
  if( mp->nrec_local > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_number_receiver_global),mp->nrec_local*sizeof(int)),1);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_number_receiver_global,h_number_receiver_global,
                                     mp->nrec_local*sizeof(int),cudaMemcpyHostToDevice),1512);
  }
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_selected_rec),nrec*sizeof(int)),1513);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_selected_rec,h_ispec_selected_rec,
                                     nrec*sizeof(int),cudaMemcpyHostToDevice),1514);

#ifdef USE_MESH_COLORING_GPU
  mp->use_mesh_coloring_gpu = 1;
  if( ! *USE_MESH_COLORING_GPU_f ) exit_on_error("error with USE_MESH_COLORING_GPU constant; please re-compile\n");
#else
  // mesh coloring
  // note: this here passes the coloring as an option to the kernel routines
  //          the performance seems to be the same if one uses the pre-processing directives above or not
  mp->use_mesh_coloring_gpu = *USE_MESH_COLORING_GPU_f;
#endif

  // number of elements per domain
  mp->nspec_acoustic = *nspec_acoustic;
  mp->nspec_elastic = *nspec_elastic;

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_constants_device");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

// purely adjoint & kernel simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_sim2_or_3_const_device,
              PREPARE_SIM2_OR_3_CONST_DEVICE)(
                                              long* Mesh_pointer_f,
                                              int* islice_selected_rec,
                                              int* islice_selected_rec_size,
                                              int* nadj_rec_local,
                                              int* nrec,
                                              int* myrank) {

TRACE("prepare_sim2_or_3_const_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  // allocates arrays for receivers
  print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_islice_selected_rec,
                                     *islice_selected_rec_size*sizeof(int)),7001);
  // copies arrays to GPU device
  print_CUDA_error_if_any(cudaMemcpy(mp->d_islice_selected_rec,islice_selected_rec,
                                     *islice_selected_rec_size*sizeof(int),cudaMemcpyHostToDevice),7002);

  // adjoint source arrays
  mp->nadj_rec_local = *nadj_rec_local;
  if( mp->nadj_rec_local > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_adj_sourcearrays,
                                       (mp->nadj_rec_local)*3*NGLL3*sizeof(realw)),7003);
                                       
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_pre_computed_irec,
                                       (mp->nadj_rec_local)*sizeof(int)),7004);

    // prepares local irec array:
    // the irec_local variable needs to be precomputed (as
    // h_pre_comp..), because normally it is in the loop updating accel,
    // and due to how it's incremented, it cannot be parallelized
    int* h_pre_computed_irec = (int*) malloc( (mp->nadj_rec_local)*sizeof(int) );
    if( h_pre_computed_irec == NULL ) exit_on_error("prepare_sim2_or_3_const_device: h_pre_computed_irec not allocated\n");

    int irec_local = 0;
    for(int irec = 0; irec < *nrec; irec++) {
      if(*myrank == islice_selected_rec[irec]) {
        irec_local++;
        h_pre_computed_irec[irec_local-1] = irec;
      }
    }
    if( irec_local != mp->nadj_rec_local ) exit_on_error("prepare_sim2_or_3_const_device: irec_local not equal\n");
    // copies values onto GPU
    print_CUDA_error_if_any(cudaMemcpy(mp->d_pre_computed_irec,h_pre_computed_irec,
                                       (mp->nadj_rec_local)*sizeof(int),cudaMemcpyHostToDevice),7010);
    free(h_pre_computed_irec);

    // temporary array to prepare extracted source array values
    mp->h_adj_sourcearrays_slice = (realw*) malloc( (mp->nadj_rec_local)*3*NGLL3*sizeof(realw) );
    if( mp->h_adj_sourcearrays_slice == NULL ) exit_on_error("h_adj_sourcearrays_slice not allocated\n");

  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_sim2_or_3_const_device");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

// for ACOUSTIC simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_fields_acoustic_device,
              PREPARE_FIELDS_ACOUSTIC_DEVICE)(long* Mesh_pointer_f,
                                              realw* rmass_acoustic,
                                              realw* rhostore,
                                              realw* kappastore,
                                              int* num_phase_ispec_acoustic,
                                              int* phase_ispec_inner_acoustic,
                                              int* ispec_is_acoustic,
                                              int* NOISE_TOMOGRAPHY,
                                              int* num_free_surface_faces,
                                              int* free_surface_ispec,
                                              int* free_surface_ijk,
                                              int* ABSORBING_CONDITIONS,
                                              int* b_reclen_potential,
                                              realw* b_absorb_potential,
                                              int* ELASTIC_SIMULATION,
                                              int* num_coupling_ac_el_faces,
                                              int* coupling_ac_el_ispec,
                                              int* coupling_ac_el_ijk,
                                              realw* coupling_ac_el_normal,
                                              realw* coupling_ac_el_jacobian2Dw,
                                              int* num_colors_outer_acoustic,
                                              int* num_colors_inner_acoustic,
                                              int* num_elem_colors_acoustic) {

  TRACE("prepare_fields_acoustic_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  /* Assuming NGLLX==5. Padded is then 128 (5^3+3) */
  int size_padded = NGLL3_PADDED * mp->NSPEC_AB;
  int size_nonpadded = NGLL3 * mp->NSPEC_AB;
  int size_glob = mp->NGLOB_AB;

  // allocates arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_acoustic),sizeof(realw)*size_glob),9001);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_dot_acoustic),sizeof(realw)*size_glob),9002);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_dot_dot_acoustic),sizeof(realw)*size_glob),9003);

  // mpi buffer
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_potential_dot_dot_buffer), 
                      (mp->max_nibool_interfaces_ext_mesh)*(mp->num_interfaces_ext_mesh)*sizeof(realw)),9004);

  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass_acoustic),sizeof(realw)*size_glob),9005);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass_acoustic,rmass_acoustic,
                                     sizeof(realw)*size_glob,cudaMemcpyHostToDevice),9100);

  // padded array
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rhostore),size_padded*sizeof(realw)),9006);
  // transfer constant element data with padding
  for(int i=0; i < mp->NSPEC_AB; i++) {
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rhostore+i*NGLL3_PADDED, &rhostore[i*NGLL3],
                                       NGLL3*sizeof(realw),cudaMemcpyHostToDevice),9106);
  }

  // non-padded array
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappastore),size_nonpadded*sizeof(realw)),9007);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_kappastore,kappastore,
                                     NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),9105);

  // phase elements
  mp->num_phase_ispec_acoustic = *num_phase_ispec_acoustic;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_phase_ispec_inner_acoustic),
                                      mp->num_phase_ispec_acoustic*2*sizeof(int)),9008);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_phase_ispec_inner_acoustic,phase_ispec_inner_acoustic,
                                     mp->num_phase_ispec_acoustic*2*sizeof(int),cudaMemcpyHostToDevice),9101);

  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_is_acoustic),
                                     mp->NSPEC_AB*sizeof(int)),9009);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_acoustic,ispec_is_acoustic,
                                     mp->NSPEC_AB*sizeof(int),cudaMemcpyHostToDevice),9102);

  // free surface
  if( *NOISE_TOMOGRAPHY == 0 ){
    // allocate surface arrays
    mp->num_free_surface_faces = *num_free_surface_faces;
    if( mp->num_free_surface_faces > 0 ){
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ispec),
                                       mp->num_free_surface_faces*sizeof(int)),9201);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec,free_surface_ispec,
                                       mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),9203);

      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ijk),
                                       3*NGLL2*mp->num_free_surface_faces*sizeof(int)),9202);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                       3*NGLL2*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),9204);
    }
  }

  // absorbing boundaries
  if( *ABSORBING_CONDITIONS ){
    mp->d_b_reclen_potential = *b_reclen_potential;
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_absorb_potential),mp->d_b_reclen_potential),9301);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_absorb_potential,b_absorb_potential,
                                       mp->d_b_reclen_potential,cudaMemcpyHostToDevice),9302);
  }


  // for seismograms
  if( mp->nrec_local > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_station_seismo_potential),
                                       mp->nrec_local*NGLL3*sizeof(realw)),9107);

    mp->h_station_seismo_potential = (realw*) malloc( mp->nrec_local*NGLL3*sizeof(realw) );
    if( mp->h_station_seismo_potential == NULL) exit_on_error("error allocating h_station_seismo_potential");
  }


  // coupling with elastic parts
  if( *ELASTIC_SIMULATION && *num_coupling_ac_el_faces > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_ispec),
                                       (*num_coupling_ac_el_faces)*sizeof(int)),9601);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_ispec,coupling_ac_el_ispec,
                                       (*num_coupling_ac_el_faces)*sizeof(int),cudaMemcpyHostToDevice),9602);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_ijk),
                                       3*NGLL2*(*num_coupling_ac_el_faces)*sizeof(int)),9603);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_ijk,coupling_ac_el_ijk,
                                       3*NGLL2*(*num_coupling_ac_el_faces)*sizeof(int),cudaMemcpyHostToDevice),9604);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_normal),
                                        3*NGLL2*(*num_coupling_ac_el_faces)*sizeof(realw)),9605);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_normal,coupling_ac_el_normal,
                                        3*NGLL2*(*num_coupling_ac_el_faces)*sizeof(realw),cudaMemcpyHostToDevice),9606);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_jacobian2Dw),
                                        NGLL2*(*num_coupling_ac_el_faces)*sizeof(realw)),9607);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_jacobian2Dw,coupling_ac_el_jacobian2Dw,
                                        NGLL2*(*num_coupling_ac_el_faces)*sizeof(realw),cudaMemcpyHostToDevice),9608);

  }

  // mesh coloring
  if( mp->use_mesh_coloring_gpu ){
    mp->num_colors_outer_acoustic = *num_colors_outer_acoustic;
    mp->num_colors_inner_acoustic = *num_colors_inner_acoustic;
    mp->h_num_elem_colors_acoustic = (int*) num_elem_colors_acoustic;
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_fields_acoustic_device");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_fields_acoustic_adj_dev,
              PREPARE_FIELDS_ACOUSTIC_ADJ_DEV)(long* Mesh_pointer_f,
                                              int* SIMULATION_TYPE,
                                              int* APPROXIMATE_HESS_KL) {

  TRACE("prepare_fields_acoustic_adj_dev");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  int size_glob = mp->NGLOB_AB;

  // kernel simulations
  if( *SIMULATION_TYPE != 3 ) return;

  // allocates backward/reconstructed arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_acoustic),sizeof(realw)*size_glob),9014);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_dot_acoustic),sizeof(realw)*size_glob),9015);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_dot_dot_acoustic),sizeof(realw)*size_glob),9016);

  // allocates kernels
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_ac_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),9017);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_ac_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),9018);

  // initializes kernel values to zero
  print_CUDA_error_if_any(cudaMemset(mp->d_rho_ac_kl,0,
                                     NGLL3*mp->NSPEC_AB*sizeof(realw)),9019);
  print_CUDA_error_if_any(cudaMemset(mp->d_kappa_ac_kl,0,
                                     NGLL3*mp->NSPEC_AB*sizeof(realw)),9020);

  // preconditioner
  if( *APPROXIMATE_HESS_KL ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_hess_ac_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),9030);
    // initializes with zeros
    print_CUDA_error_if_any(cudaMemset(mp->d_hess_ac_kl,0,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw)),9031);
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_fields_acoustic_adj_dev");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

// for ELASTIC simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_fields_elastic_device,
              PREPARE_FIELDS_ELASTIC_DEVICE)(long* Mesh_pointer_f,
                                             int* size,
                                             realw* rmass,
                                             realw* rho_vp,
                                             realw* rho_vs,
                                             int* num_phase_ispec_elastic,
                                             int* phase_ispec_inner_elastic,
                                             int* ispec_is_elastic,
                                             int* ABSORBING_CONDITIONS,
                                             realw* h_b_absorb_field,
                                             int* h_b_reclen_field,
                                             int* SIMULATION_TYPE,int* SAVE_FORWARD,
                                             int* COMPUTE_AND_STORE_STRAIN,
                                             realw* epsilondev_xx,realw* epsilondev_yy,realw* epsilondev_xy,
                                             realw* epsilondev_xz,realw* epsilondev_yz,
                                             int* ATTENUATION,
                                             int* R_size,
                                             realw* R_xx,realw* R_yy,realw* R_xy,realw* R_xz,realw* R_yz,
                                             realw* one_minus_sum_beta,realw* factor_common,
                                             realw* alphaval,realw* betaval,realw* gammaval,
                                             int* OCEANS,
                                             realw* rmass_ocean_load,
                                             int* NOISE_TOMOGRAPHY,
                                             realw* free_surface_normal,
                                             int* free_surface_ispec,
                                             int* free_surface_ijk,
                                             int* num_free_surface_faces,
                                             int* ACOUSTIC_SIMULATION,
                                             int* num_colors_outer_elastic,
                                             int* num_colors_inner_elastic,
                                             int* num_elem_colors_elastic,
                                             int* ANISOTROPY,
                                             realw *c11store,
                                             realw *c12store,
                                             realw *c13store,
                                             realw *c14store,
                                             realw *c15store,
                                             realw *c16store,
                                             realw *c22store,
                                             realw *c23store,
                                             realw *c24store,
                                             realw *c25store,
                                             realw *c26store,
                                             realw *c33store,
                                             realw *c34store,
                                             realw *c35store,
                                             realw *c36store,
                                             realw *c44store,
                                             realw *c45store,
                                             realw *c46store,
                                             realw *c55store,
                                             realw *c56store,
                                             realw *c66store){

TRACE("prepare_fields_elastic_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  /* Assuming NGLLX==5. Padded is then 128 (5^3+3) */
  int size_padded = NGLL3_PADDED * (mp->NSPEC_AB);
  int size_nonpadded = NGLL3 * (mp->NSPEC_AB);

  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_displ),sizeof(realw)*(*size)),8001);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_veloc),sizeof(realw)*(*size)),8002);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_accel),sizeof(realw)*(*size)),8003);

  // mpi buffer
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_accel_buffer),
                        3*(mp->max_nibool_interfaces_ext_mesh)*(mp->num_interfaces_ext_mesh)*sizeof(realw)),8004);

  // mass matrix
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass),sizeof(realw)*mp->NGLOB_AB),8005);
  // transfer element data
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass,rmass,
                                     sizeof(realw)*mp->NGLOB_AB,cudaMemcpyHostToDevice),8010);


  // element indices
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_ispec_is_elastic),mp->NSPEC_AB*sizeof(int)),8009);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ispec_is_elastic,ispec_is_elastic,
                                     mp->NSPEC_AB*sizeof(int),cudaMemcpyHostToDevice),8012);

  // daniel: check
  //check_ispec_is(Mesh_pointer_f,1);

  // phase elements
  mp->num_phase_ispec_elastic = *num_phase_ispec_elastic;
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_phase_ispec_inner_elastic),
                                     mp->num_phase_ispec_elastic*2*sizeof(int)),8008);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_phase_ispec_inner_elastic,phase_ispec_inner_elastic,
                                     mp->num_phase_ispec_elastic*2*sizeof(int),cudaMemcpyHostToDevice),8011);

  //daniel: check
  //check_phase_ispec(Mesh_pointer_f,1);

  // for seismograms
  if( mp->nrec_local > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_station_seismo_field),
                                     3*NGLL3*(mp->nrec_local)*sizeof(realw)),8015);

    mp->h_station_seismo_field = (realw*) malloc( 3*NGLL3*(mp->nrec_local)*sizeof(realw) );
    if( mp->h_station_seismo_field == NULL) exit_on_error("h_station_seismo_field not allocated \n");
  }

  // absorbing conditions
  if( *ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0){
    // non-padded arrays
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_vp),size_nonpadded*sizeof(realw)),8006);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_vs),size_nonpadded*sizeof(realw)),8007);

    // rho_vp, rho_vs non-padded; they are needed for stacey boundary condition
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vp, rho_vp,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),8013);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vs, rho_vs,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),8014);

    // absorb_field array used for file i/o
    if(*SIMULATION_TYPE == 3 || ( *SIMULATION_TYPE == 1 && *SAVE_FORWARD )){
      mp->d_b_reclen_field = *h_b_reclen_field;
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_absorb_field),
                                       mp->d_b_reclen_field),8016);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_b_absorb_field, h_b_absorb_field,
                                       mp->d_b_reclen_field,cudaMemcpyHostToDevice),8017);
    }
  }

  // strains used for attenuation and kernel simulations
  if( *COMPUTE_AND_STORE_STRAIN ){
    // strains
    int epsilondev_size = NGLL3*mp->NSPEC_AB; // note: non-aligned; if align, check memcpy below and indexing

    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xx,
                                       epsilondev_size*sizeof(realw)),8301);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_xx,epsilondev_xx,epsilondev_size*sizeof(realw),
                                       cudaMemcpyHostToDevice),8302);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yy,
                                       epsilondev_size*sizeof(realw)),8302);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_yy,epsilondev_yy,epsilondev_size*sizeof(realw),
                                       cudaMemcpyHostToDevice),8303);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xy,
                                       epsilondev_size*sizeof(realw)),8304);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_xy,epsilondev_xy,epsilondev_size*sizeof(realw),
                                       cudaMemcpyHostToDevice),8305);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xz,
                                       epsilondev_size*sizeof(realw)),8306);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_xz,epsilondev_xz,epsilondev_size*sizeof(realw),
                                       cudaMemcpyHostToDevice),8307);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yz,
                                       epsilondev_size*sizeof(realw)),8308);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_yz,epsilondev_yz,epsilondev_size*sizeof(realw),
                                       cudaMemcpyHostToDevice),8309);

  }

  // attenuation memory variables
  if( *ATTENUATION ){
    // memory arrays
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_xx),
                                       (*R_size)*sizeof(realw)),8401);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_xx,R_xx,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8402);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_yy),
                                       (*R_size)*sizeof(realw)),8403);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_yy,R_yy,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8404);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_xy),
                                       (*R_size)*sizeof(realw)),8405);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_xy,R_xy,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8406);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_xz),
                                       (*R_size)*sizeof(realw)),8407);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_xz,R_xz,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8408);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_yz),
                                       (*R_size)*sizeof(realw)),8409);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_yz,R_yz,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8410);

    // attenuation factors
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_one_minus_sum_beta),
                                       NGLL3*mp->NSPEC_AB*sizeof(realw)),8430);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_one_minus_sum_beta ,one_minus_sum_beta,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),8431);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_factor_common),
                                       N_SLS*NGLL3*mp->NSPEC_AB*sizeof(realw)),8432);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_factor_common ,factor_common,
                                       N_SLS*NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),8433);

    // alpha,beta,gamma factors
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_alphaval),
                                       N_SLS*sizeof(realw)),8434);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_alphaval ,alphaval,
                                       N_SLS*sizeof(realw),cudaMemcpyHostToDevice),8435);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_betaval),
                                       N_SLS*sizeof(realw)),8436);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_betaval ,betaval,
                                       N_SLS*sizeof(realw),cudaMemcpyHostToDevice),8437);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_gammaval),
                                       N_SLS*sizeof(realw)),8438);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammaval ,gammaval,
                                       N_SLS*sizeof(realw),cudaMemcpyHostToDevice),8439);

  }

  // anisotropy
  if( *ANISOTROPY ){
    // allocates memory on GPU
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c11store),
                                       size_padded*sizeof(realw)),8700);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c12store),
                                       size_padded*sizeof(realw)),8701);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c13store),
                                       size_padded*sizeof(realw)),8702);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c14store),
                                       size_padded*sizeof(realw)),8703);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c15store),
                                       size_padded*sizeof(realw)),8704);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c16store),
                                       size_padded*sizeof(realw)),8705);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c22store),
                                       size_padded*sizeof(realw)),8706);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c23store),
                                       size_padded*sizeof(realw)),8707);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c24store),
                                       size_padded*sizeof(realw)),8708);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c25store),
                                       size_padded*sizeof(realw)),8709);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c26store),
                                       size_padded*sizeof(realw)),8710);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c33store),
                                       size_padded*sizeof(realw)),8711);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c34store),
                                       size_padded*sizeof(realw)),8712);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c35store),
                                       size_padded*sizeof(realw)),8713);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c36store),
                                       size_padded*sizeof(realw)),8714);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c44store),
                                       size_padded*sizeof(realw)),8715);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c45store),
                                       size_padded*sizeof(realw)),8716);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c46store),
                                       size_padded*sizeof(realw)),8717);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c55store),
                                       size_padded*sizeof(realw)),8718);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c56store),
                                       size_padded*sizeof(realw)),8719);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_c66store),
                                       size_padded*sizeof(realw)),8720);

    // transfer constant element data with padding
    for(int i=0;i < mp->NSPEC_AB;i++) {
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c11store + i*NGLL3_PADDED, &c11store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8800);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c12store + i*NGLL3_PADDED, &c12store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8801);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c13store + i*NGLL3_PADDED, &c13store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8802);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c14store + i*NGLL3_PADDED, &c14store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8803);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c15store + i*NGLL3_PADDED, &c15store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8804);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c16store + i*NGLL3_PADDED, &c16store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8805);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c22store + i*NGLL3_PADDED, &c22store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8806);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c23store + i*NGLL3_PADDED, &c23store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8807);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c24store + i*NGLL3_PADDED, &c24store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8808);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c25store + i*NGLL3_PADDED, &c25store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8809);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c26store + i*NGLL3_PADDED, &c26store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8810);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c33store + i*NGLL3_PADDED, &c33store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8811);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c34store + i*NGLL3_PADDED, &c34store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8812);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c35store + i*NGLL3_PADDED, &c35store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8813);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c36store + i*NGLL3_PADDED, &c36store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8814);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c44store + i*NGLL3_PADDED, &c44store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8815);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c45store + i*NGLL3_PADDED, &c45store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8816);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c46store + i*NGLL3_PADDED, &c46store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8817);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c55store + i*NGLL3_PADDED, &c55store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8818);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c56store + i*NGLL3_PADDED, &c56store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8819);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_c66store + i*NGLL3_PADDED, &c66store[i*NGLL3],
                                         NGLL3*sizeof(realw),cudaMemcpyHostToDevice),8820);
    }
  }

  // ocean load approximation
  if( *OCEANS ){
    // oceans needs a free surface
    mp->num_free_surface_faces = *num_free_surface_faces;
    if( mp->num_free_surface_faces > 0 ){
      // mass matrix
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass_ocean_load),
                                         sizeof(realw)*mp->NGLOB_AB),8501);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass_ocean_load,rmass_ocean_load,
                                         sizeof(realw)*mp->NGLOB_AB,cudaMemcpyHostToDevice),8502);
      // surface normal
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_normal),
                                         3*NGLL2*(mp->num_free_surface_faces)*sizeof(realw)),8503);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_normal,free_surface_normal,
                                         3*NGLL2*(mp->num_free_surface_faces)*sizeof(realw),cudaMemcpyHostToDevice),8504);

      // temporary global array: used to synchronize updates on global accel array
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_updated_dof_ocean_load),
                                         sizeof(int)*mp->NGLOB_AB),8505);

      if( *NOISE_TOMOGRAPHY == 0 && *ACOUSTIC_SIMULATION == 0 ){
        print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ispec),
                                          mp->num_free_surface_faces*sizeof(int)),8601);
        print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec,free_surface_ispec,
                                          mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),8603);
        print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ijk),
                                          3*NGLL2*mp->num_free_surface_faces*sizeof(int)),8602);
        print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                          3*NGLL2*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),8604);
      }
    }
  }

  // mesh coloring
  if( mp->use_mesh_coloring_gpu ){
    mp->num_colors_outer_elastic = *num_colors_outer_elastic;
    mp->num_colors_inner_elastic = *num_colors_inner_elastic;
    mp->h_num_elem_colors_elastic = (int*) num_elem_colors_elastic;
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_fields_elastic_device");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_fields_elastic_adj_dev,
              PREPARE_FIELDS_ELASTIC_ADJ_DEV)(long* Mesh_pointer_f,
                                             int* size,
                                             int* SIMULATION_TYPE,
                                             int* COMPUTE_AND_STORE_STRAIN,
                                             realw* epsilon_trace_over_3,
                                             realw* b_epsilondev_xx,realw* b_epsilondev_yy,realw* b_epsilondev_xy,
                                             realw* b_epsilondev_xz,realw* b_epsilondev_yz,
                                             realw* b_epsilon_trace_over_3,
                                             int* ATTENUATION,
                                             int* R_size,
                                             realw* b_R_xx,realw* b_R_yy,realw* b_R_xy,realw* b_R_xz,realw* b_R_yz,
                                             realw* b_alphaval,realw* b_betaval,realw* b_gammaval,
                                             int* APPROXIMATE_HESS_KL){

  TRACE("prepare_fields_elastic_adj_dev");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  // checks if kernel simulation
  if( *SIMULATION_TYPE != 3 ) return;

  // kernel simulations
  // allocates backward/reconstructed arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_displ),sizeof(realw)*(*size)),8201);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_veloc),sizeof(realw)*(*size)),8202);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_accel),sizeof(realw)*(*size)),8203);

  // allocates kernels
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),8204);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_mu_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),8205);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),8206);

  // initializes kernel values to zero
  print_CUDA_error_if_any(cudaMemset(mp->d_rho_kl,0,
                                     NGLL3*mp->NSPEC_AB*sizeof(realw)),8207);
  print_CUDA_error_if_any(cudaMemset(mp->d_mu_kl,0,
                                     NGLL3*mp->NSPEC_AB*sizeof(realw)),8208);
  print_CUDA_error_if_any(cudaMemset(mp->d_kappa_kl,0,
                                     NGLL3*mp->NSPEC_AB*sizeof(realw)),8209);

  // strains used for attenuation and kernel simulations
  if( *COMPUTE_AND_STORE_STRAIN ){
    // strains
    int epsilondev_size = NGLL3*mp->NSPEC_AB; // note: non-aligned; if align, check memcpy below and indexing

    // solid pressure
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_epsilon_trace_over_3),
                                       NGLL3*mp->NSPEC_AB*sizeof(realw)),8310);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilon_trace_over_3,epsilon_trace_over_3,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),8311);
    // backward solid pressure
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilon_trace_over_3),
                                       NGLL3*mp->NSPEC_AB*sizeof(realw)),8312);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilon_trace_over_3 ,b_epsilon_trace_over_3,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw),cudaMemcpyHostToDevice),8313);
    // prepares backward strains
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xx),
                                       epsilondev_size*sizeof(realw)),8321);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_yy),
                                       epsilondev_size*sizeof(realw)),8322);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xy),
                                       epsilondev_size*sizeof(realw)),8323);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xz),
                                       epsilondev_size*sizeof(realw)),8324);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_yz),
                                       epsilondev_size*sizeof(realw)),8325);

    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_xx,b_epsilondev_xx,
                                       epsilondev_size*sizeof(realw),cudaMemcpyHostToDevice),8326);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_yy,b_epsilondev_yy,
                                       epsilondev_size*sizeof(realw),cudaMemcpyHostToDevice),8327);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_xy,b_epsilondev_xy,
                                       epsilondev_size*sizeof(realw),cudaMemcpyHostToDevice),8328);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_xz,b_epsilondev_xz,
                                       epsilondev_size*sizeof(realw),cudaMemcpyHostToDevice),8329);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_yz,b_epsilondev_yz,
                                       epsilondev_size*sizeof(realw),cudaMemcpyHostToDevice),8330);
  }

  // attenuation memory variables
  if( *ATTENUATION ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_xx),
                                       (*R_size)*sizeof(realw)),8421);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_xx,b_R_xx,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8422);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_yy),
                                       (*R_size)*sizeof(realw)),8423);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_yy,b_R_yy,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8424);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_xy),
                                       (*R_size)*sizeof(realw)),8425);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_xy,b_R_xy,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8426);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_xz),
                                       (*R_size)*sizeof(realw)),8427);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_xz,b_R_xz,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8428);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_yz),
                                       (*R_size)*sizeof(realw)),8429);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_yz,b_R_yz,(*R_size)*sizeof(realw),
                                       cudaMemcpyHostToDevice),8420);

    // alpha,beta,gamma factors for backward fields
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_alphaval),
                                       N_SLS*sizeof(realw)),8434);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_alphaval ,b_alphaval,
                                       N_SLS*sizeof(realw),cudaMemcpyHostToDevice),8435);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_betaval),
                                       N_SLS*sizeof(realw)),8436);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_betaval ,b_betaval,
                                       N_SLS*sizeof(realw),cudaMemcpyHostToDevice),8437);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_gammaval),
                                       N_SLS*sizeof(realw)),8438);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_gammaval ,b_gammaval,
                                       N_SLS*sizeof(realw),cudaMemcpyHostToDevice),8439);
  }

  if( *APPROXIMATE_HESS_KL ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_hess_el_kl),NGLL3*mp->NSPEC_AB*sizeof(realw)),8450);
    // initializes with zeros
    print_CUDA_error_if_any(cudaMemset(mp->d_hess_el_kl,0,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw)),8451);
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("prepare_fields_elastic_adj_dev");
#endif
}



/* ----------------------------------------------------------------------------------------------- */

// for NOISE simulations

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_fields_noise_device,
              PREPARE_FIELDS_NOISE_DEVICE)(long* Mesh_pointer_f,
                                           int* NSPEC_AB, int* NGLOB_AB,
                                           int* free_surface_ispec,
                                           int* free_surface_ijk,
                                           int* num_free_surface_faces,
                                           int* SIMULATION_TYPE,
                                           int* NOISE_TOMOGRAPHY,
                                           int* NSTEP,
                                           realw* noise_sourcearray,
                                           realw* normal_x_noise,
                                           realw* normal_y_noise,
                                           realw* normal_z_noise,
                                           realw* mask_noise,
                                           realw* free_surface_jacobian2Dw) {

  TRACE("prepare_fields_noise_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  // free surface
  mp->num_free_surface_faces = *num_free_surface_faces;

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ispec,
                                     mp->num_free_surface_faces*sizeof(int)),4001);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec, free_surface_ispec,
                                     mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),4002);

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ijk,
                                     3*NGLL2*mp->num_free_surface_faces*sizeof(int)),4003);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                     3*NGLL2*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),4004);

  // alloc storage for the surface buffer to be copied
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_noise_surface_movie,
                                     3*NGLL2*mp->num_free_surface_faces*sizeof(realw)),4005);

  // prepares noise source array
  if( *NOISE_TOMOGRAPHY == 1 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_noise_sourcearray,
                                       3*NGLL3*(*NSTEP)*sizeof(realw)),4101);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_noise_sourcearray, noise_sourcearray,
                                       3*NGLL3*(*NSTEP)*sizeof(realw),cudaMemcpyHostToDevice),4102);
  }

  // prepares noise directions
  if( *NOISE_TOMOGRAPHY > 1 ){
    int nface_size = NGLL2*(*num_free_surface_faces);
    // allocates memory on GPU
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_x_noise,
                                       nface_size*sizeof(realw)),4301);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_y_noise,
                                       nface_size*sizeof(realw)),4302);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_z_noise,
                                       nface_size*sizeof(realw)),4303);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_mask_noise,
                                       nface_size*sizeof(realw)),4304);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_free_surface_jacobian2Dw,
                                       nface_size*sizeof(realw)),4305);
    // transfers data onto GPU
    print_CUDA_error_if_any(cudaMemcpy(mp->d_normal_x_noise, normal_x_noise,
                                       nface_size*sizeof(realw),cudaMemcpyHostToDevice),4306);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_normal_y_noise, normal_y_noise,
                                       nface_size*sizeof(realw),cudaMemcpyHostToDevice),4307);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_normal_z_noise, normal_z_noise,
                                       nface_size*sizeof(realw),cudaMemcpyHostToDevice),4308);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_mask_noise, mask_noise,
                                       nface_size*sizeof(realw),cudaMemcpyHostToDevice),4309);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_jacobian2Dw, free_surface_jacobian2Dw,
                                       nface_size*sizeof(realw),cudaMemcpyHostToDevice),4310);
  }

  // prepares noise strength kernel
  if( *NOISE_TOMOGRAPHY == 3 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_Sigma_kl),
                                       NGLL3*(mp->NSPEC_AB)*sizeof(realw)),4401);
    // initializes kernel values to zero
    print_CUDA_error_if_any(cudaMemset(mp->d_Sigma_kl,0,
                                       NGLL3*mp->NSPEC_AB*sizeof(realw)),4403);

  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  //printf("jacobian_size = %d\n",25*(*num_free_surface_faces));
  exit_on_cuda_error("prepare_fields_noise_device");
#endif
}


/* ----------------------------------------------------------------------------------------------- */

// cleanup

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(prepare_cleanup_device,
              PREPARE_CLEANUP_DEVICE)(long* Mesh_pointer_f,
                                      int* SIMULATION_TYPE,
                                      int* SAVE_FORWARD,
                                      int* ACOUSTIC_SIMULATION,
                                      int* ELASTIC_SIMULATION,
                                      int* ABSORBING_CONDITIONS,
                                      int* NOISE_TOMOGRAPHY,
                                      int* COMPUTE_AND_STORE_STRAIN,
                                      int* ATTENUATION,
                                      int* ANISOTROPY,
                                      int* OCEANS,
                                      int* APPROXIMATE_HESS_KL) {

TRACE("prepare_cleanup_device");

  // frees allocated memory arrays
  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  // frees memory on GPU
  // mesh
  cudaFree(mp->d_xix);
  cudaFree(mp->d_xiy);
  cudaFree(mp->d_xiz);
  cudaFree(mp->d_etax);
  cudaFree(mp->d_etay);
  cudaFree(mp->d_etaz);
  cudaFree(mp->d_gammax);
  cudaFree(mp->d_gammay);
  cudaFree(mp->d_gammaz);
  cudaFree(mp->d_muv);

  // absorbing boundaries
  if( *ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0 ){
    cudaFree(mp->d_abs_boundary_ispec);
    cudaFree(mp->d_abs_boundary_ijk);
    cudaFree(mp->d_abs_boundary_normal);
    cudaFree(mp->d_abs_boundary_jacobian2Dw);
  }

  // interfaces
  cudaFree(mp->d_nibool_interfaces_ext_mesh);
  cudaFree(mp->d_ibool_interfaces_ext_mesh);

  // global indexing
  cudaFree(mp->d_ispec_is_inner);
  cudaFree(mp->d_ibool);

  // sources
  if (*SIMULATION_TYPE == 1  || *SIMULATION_TYPE == 3){
    cudaFree(mp->d_sourcearrays);
    cudaFree(mp->d_stf_pre_compute);
  }

  cudaFree(mp->d_islice_selected_source);
  cudaFree(mp->d_ispec_selected_source);

  // receivers
  if( mp->nrec_local > 0 ) cudaFree(mp->d_number_receiver_global);
  cudaFree(mp->d_ispec_selected_rec);

  // ACOUSTIC arrays
  if( *ACOUSTIC_SIMULATION ){
    cudaFree(mp->d_potential_acoustic);
    cudaFree(mp->d_potential_dot_acoustic);
    cudaFree(mp->d_potential_dot_dot_acoustic);
    cudaFree(mp->d_send_potential_dot_dot_buffer);
    cudaFree(mp->d_rmass_acoustic);
    cudaFree(mp->d_rhostore);
    cudaFree(mp->d_kappastore);
    cudaFree(mp->d_phase_ispec_inner_acoustic);
    cudaFree(mp->d_ispec_is_acoustic);

    if( *NOISE_TOMOGRAPHY == 0 ){
      cudaFree(mp->d_free_surface_ispec);
      cudaFree(mp->d_free_surface_ijk);
    }

    if( *ABSORBING_CONDITIONS ) cudaFree(mp->d_b_absorb_potential);

    if( *SIMULATION_TYPE == 3 ) {
      cudaFree(mp->d_b_potential_acoustic);
      cudaFree(mp->d_b_potential_dot_acoustic);
      cudaFree(mp->d_b_potential_dot_dot_acoustic);
      cudaFree(mp->d_rho_ac_kl);
      cudaFree(mp->d_kappa_ac_kl);
      if( *APPROXIMATE_HESS_KL) cudaFree(mp->d_hess_ac_kl);
    }


    if(mp->nrec_local > 0 ){
      cudaFree(mp->d_station_seismo_potential);
      free(mp->h_station_seismo_potential);
    }

  } // ACOUSTIC_SIMULATION

  // ELASTIC arrays
  if( *ELASTIC_SIMULATION ){
    cudaFree(mp->d_displ);
    cudaFree(mp->d_veloc);
    cudaFree(mp->d_accel);
    cudaFree(mp->d_send_accel_buffer);
    cudaFree(mp->d_rmass);

    cudaFree(mp->d_phase_ispec_inner_elastic);
    cudaFree(mp->d_ispec_is_elastic);

    if( mp->nrec_local > 0 ){
      cudaFree(mp->d_station_seismo_field);
      free(mp->h_station_seismo_field);
    }

    if( *ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0){
      cudaFree(mp->d_rho_vp);
      cudaFree(mp->d_rho_vs);

      if(*SIMULATION_TYPE == 3 || ( *SIMULATION_TYPE == 1 && *SAVE_FORWARD ))
          cudaFree(mp->d_b_absorb_field);
    }

    if( *SIMULATION_TYPE == 3 ) {
      cudaFree(mp->d_b_displ);
      cudaFree(mp->d_b_veloc);
      cudaFree(mp->d_b_accel);
      cudaFree(mp->d_rho_kl);
      cudaFree(mp->d_mu_kl);
      cudaFree(mp->d_kappa_kl);
      if( *APPROXIMATE_HESS_KL ) cudaFree(mp->d_hess_el_kl);
    }

    if( *COMPUTE_AND_STORE_STRAIN ){
      cudaFree(mp->d_epsilondev_xx);
      cudaFree(mp->d_epsilondev_yy);
      cudaFree(mp->d_epsilondev_xy);
      cudaFree(mp->d_epsilondev_xz);
      cudaFree(mp->d_epsilondev_yz);
      if( *SIMULATION_TYPE == 3 ){
        cudaFree(mp->d_epsilon_trace_over_3);
        cudaFree(mp->d_b_epsilon_trace_over_3);
        cudaFree(mp->d_b_epsilondev_xx);
        cudaFree(mp->d_b_epsilondev_yy);
        cudaFree(mp->d_b_epsilondev_xy);
        cudaFree(mp->d_b_epsilondev_xz);
        cudaFree(mp->d_b_epsilondev_yz);
      }
    }

    if( *ATTENUATION ){
      cudaFree(mp->d_factor_common);
      cudaFree(mp->d_one_minus_sum_beta);
      cudaFree(mp->d_alphaval);
      cudaFree(mp->d_betaval);
      cudaFree(mp->d_gammaval);
      cudaFree(mp->d_R_xx);
      cudaFree(mp->d_R_yy);
      cudaFree(mp->d_R_xy);
      cudaFree(mp->d_R_xz);
      cudaFree(mp->d_R_yz);
      if( *SIMULATION_TYPE == 3){
        cudaFree(mp->d_b_R_xx);
        cudaFree(mp->d_b_R_yy);
        cudaFree(mp->d_b_R_xy);
        cudaFree(mp->d_b_R_xz);
        cudaFree(mp->d_b_R_yz);
        cudaFree(mp->d_b_alphaval);
        cudaFree(mp->d_b_betaval);
        cudaFree(mp->d_b_gammaval);
      }
    }

    if( *ANISOTROPY ){
      cudaFree(mp->d_c11store);
      cudaFree(mp->d_c12store);
      cudaFree(mp->d_c13store);
      cudaFree(mp->d_c14store);
      cudaFree(mp->d_c15store);
      cudaFree(mp->d_c16store);
      cudaFree(mp->d_c22store);
      cudaFree(mp->d_c23store);
      cudaFree(mp->d_c24store);
      cudaFree(mp->d_c25store);
      cudaFree(mp->d_c26store);
      cudaFree(mp->d_c33store);
      cudaFree(mp->d_c34store);
      cudaFree(mp->d_c35store);
      cudaFree(mp->d_c36store);
      cudaFree(mp->d_c44store);
      cudaFree(mp->d_c45store);
      cudaFree(mp->d_c46store);
      cudaFree(mp->d_c55store);
      cudaFree(mp->d_c56store);
      cudaFree(mp->d_c66store);
    }

    if( *OCEANS ){
      if( mp->num_free_surface_faces > 0 ){
        cudaFree(mp->d_rmass_ocean_load);
        cudaFree(mp->d_free_surface_normal);
        cudaFree(mp->d_updated_dof_ocean_load);
        if( *NOISE_TOMOGRAPHY == 0){
          cudaFree(mp->d_free_surface_ispec);
          cudaFree(mp->d_free_surface_ijk);
        }
      }
    }
  } // ELASTIC_SIMULATION

  // purely adjoint & kernel array
  if( *SIMULATION_TYPE == 2 || *SIMULATION_TYPE == 3 ){
    cudaFree(mp->d_islice_selected_rec);
    if(mp->nadj_rec_local > 0 ){
      cudaFree(mp->d_adj_sourcearrays);
      cudaFree(mp->d_pre_computed_irec);
      free(mp->h_adj_sourcearrays_slice);
    }
  }

  // NOISE arrays
  if( *NOISE_TOMOGRAPHY > 0 ){
    cudaFree(mp->d_free_surface_ispec);
    cudaFree(mp->d_free_surface_ijk);
    cudaFree(mp->d_noise_surface_movie);
    if( *NOISE_TOMOGRAPHY == 1 ) cudaFree(mp->d_noise_sourcearray);
    if( *NOISE_TOMOGRAPHY > 1 ){
      cudaFree(mp->d_normal_x_noise);
      cudaFree(mp->d_normal_y_noise);
      cudaFree(mp->d_normal_z_noise);
      cudaFree(mp->d_mask_noise);
      cudaFree(mp->d_free_surface_jacobian2Dw);
    }
    if( *NOISE_TOMOGRAPHY == 3 ) cudaFree(mp->d_Sigma_kl);
  }

  // mesh pointer - not needed anymore
  free(mp);
}

