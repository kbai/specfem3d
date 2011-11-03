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
                                        float* h_xix, float* h_xiy, float* h_xiz,
                                        float* h_etax, float* h_etay, float* h_etaz,
                                        float* h_gammax, float* h_gammay, float* h_gammaz,
                                        float* h_kappav, float* h_muv,
                                        int* h_ibool,
                                        int* num_interfaces_ext_mesh,
                                        int* max_nibool_interfaces_ext_mesh,
                                        int* h_nibool_interfaces_ext_mesh,
                                        int* h_ibool_interfaces_ext_mesh,
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
                                        int* nsources_local_f,
                                        float* h_sourcearrays,
                                        int* h_islice_selected_source,
                                        int* h_ispec_selected_source,
                                        int* h_number_receiver_global,
                                        int* h_ispec_selected_rec,
                                        int* nrec_f,
                                        int* nrec_local_f,
                                        int* SIMULATION_TYPE,
                                        int* USE_MESH_COLORING_GPU_f,
                                        int* nspec_acoustic,int* nspec_elastic,
                                        int* ncuda_devices) {

TRACE("prepare_constants_device");

  int procid;
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
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  // Sets the active device
  if(device_count > 1) {
    // generalized for more GPUs per node
    cudaSetDevice((procid)%device_count);
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
  int size_padded = 128 * (mp->NSPEC_AB);
  int size = 125 * (mp->NSPEC_AB);

  // mesh
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xix, size_padded*sizeof(float)),1001);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xiy, size_padded*sizeof(float)),1002);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_xiz, size_padded*sizeof(float)),1003);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etax, size_padded*sizeof(float)),1004);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etay, size_padded*sizeof(float)),1005);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_etaz, size_padded*sizeof(float)),1006);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammax, size_padded*sizeof(float)),1007);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammay, size_padded*sizeof(float)),1008);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_gammaz, size_padded*sizeof(float)),1009);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_kappav, size_padded*sizeof(float)),1010);
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_muv, size_padded*sizeof(float)),1011);

  // transfer constant element data with padding
  for(int i=0;i < mp->NSPEC_AB;i++) {
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xix + i*128, &h_xix[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1501);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiy+i*128,   &h_xiy[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1502);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_xiz+i*128,   &h_xiz[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1503);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etax+i*128,  &h_etax[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1504);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etay+i*128,  &h_etay[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1505);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_etaz+i*128,  &h_etaz[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1506);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammax+i*128,&h_gammax[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1507);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammay+i*128,&h_gammay[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1508);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammaz+i*128,&h_gammaz[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1509);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_kappav+i*128,&h_kappav[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1510);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_muv+i*128,   &h_muv[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),1511);
  }

  // global indexing
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool,size_padded*sizeof(int)),1021);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool, h_ibool,
                                     size*sizeof(int),cudaMemcpyHostToDevice),1022);


  // prepare interprocess-edge exchange information
  if( *num_interfaces_ext_mesh > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_nibool_interfaces_ext_mesh,
                                       (*num_interfaces_ext_mesh)*sizeof(int)),1201);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
                                       (*num_interfaces_ext_mesh)*sizeof(int),cudaMemcpyHostToDevice),1202);

    print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_ibool_interfaces_ext_mesh,
                                       (*num_interfaces_ext_mesh)*(*max_nibool_interfaces_ext_mesh)*sizeof(int)),1203);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
                                       (*num_interfaces_ext_mesh)*(*max_nibool_interfaces_ext_mesh)*sizeof(int),
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
                                       3*25*(mp->d_num_abs_boundary_faces)*sizeof(int)),1103);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_ijk, h_abs_boundary_ijk,
                                       3*25*(mp->d_num_abs_boundary_faces)*sizeof(int),
                                       cudaMemcpyHostToDevice),1104);

    print_CUDA_error_if_any(cudaMalloc((void**) &(mp->d_abs_boundary_normal),
                                       3*25*(mp->d_num_abs_boundary_faces)*sizeof(float)),1105);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_normal, h_abs_boundary_normal,
                                       3*25*(mp->d_num_abs_boundary_faces)*sizeof(float),
                                       cudaMemcpyHostToDevice),1106);

    print_CUDA_error_if_any(cudaMalloc((void**) &(mp->d_abs_boundary_jacobian2Dw),
                                       25*(mp->d_num_abs_boundary_faces)*sizeof(float)),1107);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_abs_boundary_jacobian2Dw, h_abs_boundary_jacobian2Dw,
                                       25*(mp->d_num_abs_boundary_faces)*sizeof(float),
                                       cudaMemcpyHostToDevice),1108);
  }

  // sources
  mp->nsources_local = *nsources_local_f;
  if (*SIMULATION_TYPE == 1  || *SIMULATION_TYPE == 3){
    // not needed in case of pure adjoint simulations (SIMULATION_TYPE == 2)
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_sourcearrays,
                                       sizeof(float)* *NSOURCES*3*125),1301);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_sourcearrays, h_sourcearrays,
                                       sizeof(float)* *NSOURCES*3*125,cudaMemcpyHostToDevice),1302);

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
                                       (mp->nadj_rec_local)*3*125*sizeof(float)),7003);
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
    mp->h_adj_sourcearrays_slice = (float*) malloc( (mp->nadj_rec_local)*3*125*sizeof(float) );
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
                                              int* ELASTIC_SIMULATION,
                                              int* num_coupling_ac_el_faces,
                                              int* coupling_ac_el_ispec,
                                              int* coupling_ac_el_ijk,
                                              float* coupling_ac_el_normal,
                                              float* coupling_ac_el_jacobian2Dw,
                                              int* num_colors_outer_acoustic,
                                              int* num_colors_inner_acoustic,
                                              int* num_elem_colors_acoustic) {

  TRACE("prepare_fields_acoustic_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  /* Assuming NGLLX==5. Padded is then 128 (5^3+3) */
  int size_padded = 128 * mp->NSPEC_AB;
  int size_nonpadded = 125 * mp->NSPEC_AB;
  int size = mp->NGLOB_AB;

  // allocates arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_acoustic),sizeof(float)*size),9001);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_dot_acoustic),sizeof(float)*size),9002);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_potential_dot_dot_acoustic),sizeof(float)*size),9003);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_potential_dot_dot_buffer),sizeof(float)*size),9004);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass_acoustic),sizeof(float)*size),9005);
  // padded array
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rhostore),size_padded*sizeof(float)),9006);
  // non-padded array
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappastore),size_nonpadded*sizeof(float)),9007);

  // transfer element data
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass_acoustic,rmass_acoustic,
                                     sizeof(float)*size,cudaMemcpyHostToDevice),9100);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_kappastore,kappastore,
                                     size_nonpadded*sizeof(float),cudaMemcpyHostToDevice),9105);
  // transfer constant element data with padding
  for(int i=0; i < mp->NSPEC_AB; i++) {
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rhostore+i*128, &rhostore[i*125],
                                       125*sizeof(float),cudaMemcpyHostToDevice),9106);
  }

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
                                       3*25*mp->num_free_surface_faces*sizeof(int)),9202);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                       3*25*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),9204);
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
                                       mp->nrec_local*125*sizeof(float)),9107);
    mp->h_station_seismo_potential = (float*) malloc( mp->nrec_local*125*sizeof(float) );
    if( mp->h_station_seismo_potential == NULL) exit_on_error("error allocating h_station_seismo_potential");
  }


  // coupling with elastic parts
  if( *ELASTIC_SIMULATION && *num_coupling_ac_el_faces > 0 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_ispec),
                                       (*num_coupling_ac_el_faces)*sizeof(int)),9601);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_ispec,coupling_ac_el_ispec,
                                       (*num_coupling_ac_el_faces)*sizeof(int),cudaMemcpyHostToDevice),9602);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_ijk),
                                       3*25*(*num_coupling_ac_el_faces)*sizeof(int)),9603);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_ijk,coupling_ac_el_ijk,
                                       3*25*(*num_coupling_ac_el_faces)*sizeof(int),cudaMemcpyHostToDevice),9604);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_normal),
                                        3*25*(*num_coupling_ac_el_faces)*sizeof(float)),9605);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_normal,coupling_ac_el_normal,
                                        3*25*(*num_coupling_ac_el_faces)*sizeof(float),cudaMemcpyHostToDevice),9606);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_coupling_ac_el_jacobian2Dw),
                                        25*(*num_coupling_ac_el_faces)*sizeof(float)),9607);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_coupling_ac_el_jacobian2Dw,coupling_ac_el_jacobian2Dw,
                                        25*(*num_coupling_ac_el_faces)*sizeof(float),cudaMemcpyHostToDevice),9608);

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

  int size = mp->NGLOB_AB;

  // kernel simulations
  if( *SIMULATION_TYPE != 3 ) return;

  // allocates backward/reconstructed arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_acoustic),sizeof(float)*size),9014);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_dot_acoustic),sizeof(float)*size),9015);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_potential_dot_dot_acoustic),sizeof(float)*size),9016);

  // allocates kernels
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_ac_kl),125*mp->NSPEC_AB*sizeof(float)),9017);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_ac_kl),125*mp->NSPEC_AB*sizeof(float)),9018);

  // initializes kernel values to zero
  print_CUDA_error_if_any(cudaMemset(mp->d_rho_ac_kl,0,
                                     125*mp->NSPEC_AB*sizeof(float)),9019);
  print_CUDA_error_if_any(cudaMemset(mp->d_kappa_ac_kl,0,
                                     125*mp->NSPEC_AB*sizeof(float)),9020);

  // preconditioner
  if( *APPROXIMATE_HESS_KL ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_hess_ac_kl),125*mp->NSPEC_AB*sizeof(float)),9030);
    // initializes with zeros
    print_CUDA_error_if_any(cudaMemset(mp->d_hess_ac_kl,0,
                                       125*mp->NSPEC_AB*sizeof(float)),9031);
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
                                             float* rmass,
                                             float* rho_vp,
                                             float* rho_vs,
                                             int* num_phase_ispec_elastic,
                                             int* phase_ispec_inner_elastic,
                                             int* ispec_is_elastic,
                                             int* ABSORBING_CONDITIONS,
                                             float* h_b_absorb_field,
                                             int* h_b_reclen_field,
                                             int* SIMULATION_TYPE,int* SAVE_FORWARD,
                                             int* COMPUTE_AND_STORE_STRAIN,
                                             float* epsilondev_xx,float* epsilondev_yy,float* epsilondev_xy,
                                             float* epsilondev_xz,float* epsilondev_yz,
                                             int* ATTENUATION,
                                             int* R_size,
                                             float* R_xx,float* R_yy,float* R_xy,float* R_xz,float* R_yz,
                                             float* one_minus_sum_beta,float* factor_common,
                                             float* alphaval,float* betaval,float* gammaval,
                                             int* OCEANS,
                                             float* rmass_ocean_load,
                                             int* NOISE_TOMOGRAPHY,
                                             float* free_surface_normal,
                                             int* free_surface_ispec,
                                             int* free_surface_ijk,
                                             int* num_free_surface_faces,
                                             int* ACOUSTIC_SIMULATION,
                                             int* num_colors_outer_elastic,
                                             int* num_colors_inner_elastic,
                                             int* num_elem_colors_elastic){

TRACE("prepare_fields_elastic_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);
  /* Assuming NGLLX==5. Padded is then 128 (5^3+3) */
  //int size_padded = 128 * mp->NSPEC_AB;
  int size_nonpadded = 125 * mp->NSPEC_AB;

  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_displ),sizeof(float)*(*size)),8001);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_veloc),sizeof(float)*(*size)),8002);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_accel),sizeof(float)*(*size)),8003);

  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_send_accel_buffer),sizeof(float)*(*size)),8004);

  // mass matrix
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass),sizeof(float)*mp->NGLOB_AB),8005);
  // transfer element data
  print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass,rmass,
                                     sizeof(float)*mp->NGLOB_AB,cudaMemcpyHostToDevice),8010);


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
                                     3*125*(mp->nrec_local)*sizeof(float)),8015);
    mp->h_station_seismo_field = (float*) malloc( 3*125*(mp->nrec_local)*sizeof(float) );
    if( mp->h_station_seismo_field == NULL) exit_on_error("h_station_seismo_field not allocated \n");
  }

  // absorbing conditions
  if( *ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0){
    // non-padded arrays
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_vp),size_nonpadded*sizeof(float)),8006);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_vs),size_nonpadded*sizeof(float)),8007);

    // rho_vp, rho_vs non-padded; they are needed for stacey boundary condition
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vp, rho_vp,
                                       size_nonpadded*sizeof(float),cudaMemcpyHostToDevice),8013);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_rho_vs, rho_vs,
                                       size_nonpadded*sizeof(float),cudaMemcpyHostToDevice),8014);

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
    int epsilondev_size = 125*mp->NSPEC_AB; // note: non-aligned; if align, check memcpy below and indexing

    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xx,
                                       epsilondev_size*sizeof(float)),8301);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_xx,epsilondev_xx,epsilondev_size*sizeof(float),
                                       cudaMemcpyHostToDevice),8302);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yy,
                                       epsilondev_size*sizeof(float)),8302);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_yy,epsilondev_yy,epsilondev_size*sizeof(float),
                                       cudaMemcpyHostToDevice),8303);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xy,
                                       epsilondev_size*sizeof(float)),8304);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_xy,epsilondev_xy,epsilondev_size*sizeof(float),
                                       cudaMemcpyHostToDevice),8305);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_xz,
                                       epsilondev_size*sizeof(float)),8306);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_xz,epsilondev_xz,epsilondev_size*sizeof(float),
                                       cudaMemcpyHostToDevice),8307);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_epsilondev_yz,
                                       epsilondev_size*sizeof(float)),8308);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilondev_yz,epsilondev_yz,epsilondev_size*sizeof(float),
                                       cudaMemcpyHostToDevice),8309);

  }

  // attenuation memory variables
  if( *ATTENUATION ){
    // memory arrays
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_xx),
                                       (*R_size)*sizeof(float)),8401);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_xx,R_xx,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8402);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_yy),
                                       (*R_size)*sizeof(float)),8403);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_yy,R_yy,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8404);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_xy),
                                       (*R_size)*sizeof(float)),8405);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_xy,R_xy,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8406);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_xz),
                                       (*R_size)*sizeof(float)),8407);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_xz,R_xz,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8408);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_R_yz),
                                       (*R_size)*sizeof(float)),8409);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_R_yz,R_yz,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8410);

    // attenuation factors
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_one_minus_sum_beta),
                                       125*mp->NSPEC_AB*sizeof(float)),8430);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_one_minus_sum_beta ,one_minus_sum_beta,
                                       125*mp->NSPEC_AB*sizeof(float),cudaMemcpyHostToDevice),8431);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_factor_common),
                                       N_SLS*125*mp->NSPEC_AB*sizeof(float)),8432);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_factor_common ,factor_common,
                                       N_SLS*125*mp->NSPEC_AB*sizeof(float),cudaMemcpyHostToDevice),8433);

    // alpha,beta,gamma factors
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_alphaval),
                                       N_SLS*sizeof(float)),8434);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_alphaval ,alphaval,
                                       N_SLS*sizeof(float),cudaMemcpyHostToDevice),8435);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_betaval),
                                       N_SLS*sizeof(float)),8436);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_betaval ,betaval,
                                       N_SLS*sizeof(float),cudaMemcpyHostToDevice),8437);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_gammaval),
                                       N_SLS*sizeof(float)),8438);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_gammaval ,gammaval,
                                       N_SLS*sizeof(float),cudaMemcpyHostToDevice),8439);

  }


  if( *OCEANS ){
    // oceans needs a free surface
    mp->num_free_surface_faces = *num_free_surface_faces;
    if( mp->num_free_surface_faces > 0 ){
      // mass matrix
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rmass_ocean_load),
                                         sizeof(float)*mp->NGLOB_AB),8501);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_rmass_ocean_load,rmass_ocean_load,
                                         sizeof(float)*mp->NGLOB_AB,cudaMemcpyHostToDevice),8502);
      // surface normal
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_normal),
                                         3*25*(mp->num_free_surface_faces)*sizeof(float)),8503);
      print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_normal,free_surface_normal,
                                         3*25*(mp->num_free_surface_faces)*sizeof(float),cudaMemcpyHostToDevice),8504);

      // temporary global array: used to synchronize updates on global accel array
      print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_updated_dof_ocean_load),
                                         sizeof(int)*mp->NGLOB_AB),8505);

      if( *NOISE_TOMOGRAPHY == 0 && *ACOUSTIC_SIMULATION == 0 ){
        print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ispec),
                                          mp->num_free_surface_faces*sizeof(int)),9201);
        print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec,free_surface_ispec,
                                          mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),9203);
        print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_free_surface_ijk),
                                          3*25*mp->num_free_surface_faces*sizeof(int)),9202);
        print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                          3*25*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),9204);
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
                                             float* epsilon_trace_over_3,
                                             float* b_epsilondev_xx,float* b_epsilondev_yy,float* b_epsilondev_xy,
                                             float* b_epsilondev_xz,float* b_epsilondev_yz,
                                             float* b_epsilon_trace_over_3,
                                             int* ATTENUATION,
                                             int* R_size,
                                             float* b_R_xx,float* b_R_yy,float* b_R_xy,float* b_R_xz,float* b_R_yz,
                                             float* b_alphaval,float* b_betaval,float* b_gammaval,
                                             int* APPROXIMATE_HESS_KL){

  TRACE("prepare_fields_elastic_adj_dev");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  // checks if kernel simulation
  if( *SIMULATION_TYPE != 3 ) return;

  // kernel simulations
  // allocates backward/reconstructed arrays on device (GPU)
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_displ),sizeof(float)*(*size)),8201);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_veloc),sizeof(float)*(*size)),8202);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_accel),sizeof(float)*(*size)),8203);

  // allocates kernels
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_rho_kl),125*mp->NSPEC_AB*sizeof(float)),8204);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_mu_kl),125*mp->NSPEC_AB*sizeof(float)),8205);
  print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_kappa_kl),125*mp->NSPEC_AB*sizeof(float)),8206);

  // initializes kernel values to zero
  print_CUDA_error_if_any(cudaMemset(mp->d_rho_kl,0,
                                     125*mp->NSPEC_AB*sizeof(float)),8207);
  print_CUDA_error_if_any(cudaMemset(mp->d_mu_kl,0,
                                     125*mp->NSPEC_AB*sizeof(float)),8208);
  print_CUDA_error_if_any(cudaMemset(mp->d_kappa_kl,0,
                                     125*mp->NSPEC_AB*sizeof(float)),8209);

  // strains used for attenuation and kernel simulations
  if( *COMPUTE_AND_STORE_STRAIN ){
    // strains
    int epsilondev_size = 125*mp->NSPEC_AB; // note: non-aligned; if align, check memcpy below and indexing

    // solid pressure
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_epsilon_trace_over_3),
                                       125*mp->NSPEC_AB*sizeof(float)),8310);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_epsilon_trace_over_3,epsilon_trace_over_3,
                                       125*mp->NSPEC_AB*sizeof(float),cudaMemcpyHostToDevice),8311);
    // backward solid pressure
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilon_trace_over_3),
                                       125*mp->NSPEC_AB*sizeof(float)),8312);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilon_trace_over_3 ,b_epsilon_trace_over_3,
                                       125*mp->NSPEC_AB*sizeof(float),cudaMemcpyHostToDevice),8313);
    // prepares backward strains
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xx),
                                       epsilondev_size*sizeof(float)),8321);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_yy),
                                       epsilondev_size*sizeof(float)),8322);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xy),
                                       epsilondev_size*sizeof(float)),8323);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_xz),
                                       epsilondev_size*sizeof(float)),8324);
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_epsilondev_yz),
                                       epsilondev_size*sizeof(float)),8325);

    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_xx,b_epsilondev_xx,
                                       epsilondev_size*sizeof(float),cudaMemcpyHostToDevice),8326);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_yy,b_epsilondev_yy,
                                       epsilondev_size*sizeof(float),cudaMemcpyHostToDevice),8327);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_xy,b_epsilondev_xy,
                                       epsilondev_size*sizeof(float),cudaMemcpyHostToDevice),8328);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_xz,b_epsilondev_xz,
                                       epsilondev_size*sizeof(float),cudaMemcpyHostToDevice),8329);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_epsilondev_yz,b_epsilondev_yz,
                                       epsilondev_size*sizeof(float),cudaMemcpyHostToDevice),8330);
  }

  // attenuation memory variables
  if( *ATTENUATION ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_xx),
                                       (*R_size)*sizeof(float)),8421);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_xx,b_R_xx,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8422);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_yy),
                                       (*R_size)*sizeof(float)),8423);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_yy,b_R_yy,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8424);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_xy),
                                       (*R_size)*sizeof(float)),8425);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_xy,b_R_xy,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8426);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_xz),
                                       (*R_size)*sizeof(float)),8427);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_xz,b_R_xz,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8428);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_R_yz),
                                       (*R_size)*sizeof(float)),8429);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_R_yz,b_R_yz,(*R_size)*sizeof(float),
                                       cudaMemcpyHostToDevice),8420);

    // alpha,beta,gamma factors for backward fields
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_alphaval),
                                       N_SLS*sizeof(float)),8434);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_alphaval ,b_alphaval,
                                       N_SLS*sizeof(float),cudaMemcpyHostToDevice),8435);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_betaval),
                                       N_SLS*sizeof(float)),8436);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_betaval ,b_betaval,
                                       N_SLS*sizeof(float),cudaMemcpyHostToDevice),8437);

    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_b_gammaval),
                                       N_SLS*sizeof(float)),8438);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_gammaval ,b_gammaval,
                                       N_SLS*sizeof(float),cudaMemcpyHostToDevice),8439);
  }

  if( *APPROXIMATE_HESS_KL ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_hess_el_kl),125*mp->NSPEC_AB*sizeof(float)),8450);
    // initializes with zeros
    print_CUDA_error_if_any(cudaMemset(mp->d_hess_el_kl,0,
                                       125*mp->NSPEC_AB*sizeof(float)),8451);
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
                                           float* noise_sourcearray,
                                           float* normal_x_noise,
                                           float* normal_y_noise,
                                           float* normal_z_noise,
                                           float* mask_noise,
                                           float* free_surface_jacobian2Dw) {

  TRACE("prepare_fields_noise_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f);

  // free surface
  mp->num_free_surface_faces = *num_free_surface_faces;

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ispec,
                                     mp->num_free_surface_faces*sizeof(int)),4001);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ispec, free_surface_ispec,
                                     mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),4002);

  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_free_surface_ijk,
                                     3*25*mp->num_free_surface_faces*sizeof(int)),4003);
  print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_ijk,free_surface_ijk,
                                     3*25*mp->num_free_surface_faces*sizeof(int),cudaMemcpyHostToDevice),4004);

  // alloc storage for the surface buffer to be copied
  print_CUDA_error_if_any(cudaMalloc((void**) &mp->d_noise_surface_movie,
                                     3*25*mp->num_free_surface_faces*sizeof(float)),4005);

  // prepares noise source array
  if( *NOISE_TOMOGRAPHY == 1 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_noise_sourcearray,
                                       3*125*(*NSTEP)*sizeof(float)),4101);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_noise_sourcearray, noise_sourcearray,
                                       3*125*(*NSTEP)*sizeof(float),cudaMemcpyHostToDevice),4102);
  }

  // prepares noise directions
  if( *NOISE_TOMOGRAPHY > 1 ){
    int nface_size = 25*(*num_free_surface_faces);
    // allocates memory on GPU
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_x_noise,
                                       nface_size*sizeof(float)),4301);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_y_noise,
                                       nface_size*sizeof(float)),4302);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_normal_z_noise,
                                       nface_size*sizeof(float)),4303);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_mask_noise,
                                       nface_size*sizeof(float)),4304);
    print_CUDA_error_if_any(cudaMalloc((void**)&mp->d_free_surface_jacobian2Dw,
                                       nface_size*sizeof(float)),4305);
    // transfers data onto GPU
    print_CUDA_error_if_any(cudaMemcpy(mp->d_normal_x_noise, normal_x_noise,
                                       nface_size*sizeof(float),cudaMemcpyHostToDevice),4306);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_normal_y_noise, normal_y_noise,
                                       nface_size*sizeof(float),cudaMemcpyHostToDevice),4307);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_normal_z_noise, normal_z_noise,
                                       nface_size*sizeof(float),cudaMemcpyHostToDevice),4308);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_mask_noise, mask_noise,
                                       nface_size*sizeof(float),cudaMemcpyHostToDevice),4309);
    print_CUDA_error_if_any(cudaMemcpy(mp->d_free_surface_jacobian2Dw, free_surface_jacobian2Dw,
                                       nface_size*sizeof(float),cudaMemcpyHostToDevice),4310);
  }

  // prepares noise strength kernel
  if( *NOISE_TOMOGRAPHY == 3 ){
    print_CUDA_error_if_any(cudaMalloc((void**)&(mp->d_Sigma_kl),
                                       125*(mp->NSPEC_AB)*sizeof(float)),4401);
    // initializes kernel values to zero
    print_CUDA_error_if_any(cudaMemset(mp->d_Sigma_kl,0,
                                       125*mp->NSPEC_AB*sizeof(float)),4403);

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

