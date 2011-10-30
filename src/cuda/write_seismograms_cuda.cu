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
#include <sys/types.h>
#include <unistd.h>

#include "config.h"
#include "mesh_constants_cuda.h"


/* ----------------------------------------------------------------------------------------------- */

// ELASTIC simulations

/* ----------------------------------------------------------------------------------------------- */

__global__ void transfer_stations_fields_from_device_kernel(int* number_receiver_global,
                                                            int* ispec_selected_rec,
                                                            int* ibool,
                                                            float* station_seismo_field,
                                                            float* desired_field,
                                                            int nrec_local,int* debug_index) {
  int blockID = blockIdx.x + blockIdx.y*gridDim.x;
  if(blockID<nrec_local) {
    //int nodeID = threadIdx.x + blockID*blockDim.x;
    int irec = number_receiver_global[blockID]-1;
    int ispec = ispec_selected_rec[irec]-1; // ispec==0 before -1???
    // if(threadIdx.x==1 && blockID < 125) {
    //   // debug_index[threadIdx.x] = threadIdx.x + 125*ispec;
    //   debug_index[blockID] = ispec;
    //   debug_index[blockID + 4] = irec;
    //   debug_index[blockID + 8] = ispec_selected_rec[0];
    //   debug_index[blockID + 9] = ispec_selected_rec[1];
    //   debug_index[blockID +10] = ispec_selected_rec[2];
    //   debug_index[blockID +11] = ispec_selected_rec[3];
    //   debug_index[blockID +12] = ispec_selected_rec[4];
    // }
    int iglob = ibool[threadIdx.x + 125*ispec]-1;
    station_seismo_field[3*125*blockID + 3*threadIdx.x+0] = desired_field[3*iglob];
    station_seismo_field[3*125*blockID + 3*threadIdx.x+1] = desired_field[3*iglob+1];
    station_seismo_field[3*125*blockID + 3*threadIdx.x+2] = desired_field[3*iglob+2];
  }
}


/* ----------------------------------------------------------------------------------------------- */

void transfer_field_from_device(Mesh* mp, float* d_field,float* h_field,
                                          int* number_receiver_global,
                                          int* d_ispec_selected,
                                          int* h_ispec_selected,
                                          int* ibool) {

TRACE("transfer_field_from_device");

  // checks if anything to do
  if( mp->nrec_local == 0 ) return;

  int blocksize = 125;
  int num_blocks_x = mp->nrec_local;
  int num_blocks_y = 1;
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(blocksize,1,1);

  int* d_debug_index;
  //int* h_debug_index;
  //cudaMalloc((void**)&d_debug_index,125*sizeof(int));
  //h_debug_index = (int*)calloc(125,sizeof(int));
  //cudaMemcpy(d_debug_index,h_debug_index,125*sizeof(int),cudaMemcpyHostToDevice);


  // prepare field transfer array on device
  transfer_stations_fields_from_device_kernel<<<grid,threads>>>(mp->d_number_receiver_global,
                d_ispec_selected,
                mp->d_ibool,
                mp->d_station_seismo_field,
                d_field,
                mp->nrec_local,d_debug_index);

  //cudaMemcpy(h_debug_index,d_debug_index,125*sizeof(int),cudaMemcpyDeviceToHost);

  // pause_for_debug(1);
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("transfer_stations_fields_from_device_kernel");
#endif

  cudaMemcpy(mp->h_station_seismo_field,mp->d_station_seismo_field,
       (3*125)*(mp->nrec_local)*sizeof(float),cudaMemcpyDeviceToHost);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("transfer_stations_fields_from_device_kernel_memcpy");
#endif

  // pause_for_debug(1);
  int irec_local;

  for(irec_local=0;irec_local<mp->nrec_local;irec_local++) {
    int irec = number_receiver_global[irec_local] - 1;
    int ispec = h_ispec_selected[irec] - 1;

    for(int i=0;i<125;i++) {
      int iglob = ibool[i+125*ispec] - 1;
      h_field[0+3*iglob] = mp->h_station_seismo_field[0+3*i+irec_local*125*3];
      h_field[1+3*iglob] = mp->h_station_seismo_field[1+3*i+irec_local*125*3];
      h_field[2+3*iglob] = mp->h_station_seismo_field[2+3*i+irec_local*125*3];
    }

  }
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_station_el_from_device,
              TRANSFER_STATION_EL_FROM_DEVICE)(float* displ,float* veloc,float* accel,
                                                   float* b_displ, float* b_veloc, float* b_accel,
                                                   long* Mesh_pointer_f,int* number_receiver_global,
                                                   int* ispec_selected_rec,int* ispec_selected_source,
                                                   int* ibool,int* SIMULATION_TYPEf) {
TRACE("transfer_station_el_from_device");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper
  // checks if anything to do
  if( mp->nrec_local == 0 ) return;

  int SIMULATION_TYPE = *SIMULATION_TYPEf;

  if(SIMULATION_TYPE == 1) {
    transfer_field_from_device(mp,mp->d_displ,displ, number_receiver_global,
             mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_from_device(mp,mp->d_veloc,veloc, number_receiver_global,
             mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_from_device(mp,mp->d_accel,accel, number_receiver_global,
             mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
  }
  else if(SIMULATION_TYPE == 2) {
    transfer_field_from_device(mp,mp->d_displ,displ, number_receiver_global,
             mp->d_ispec_selected_source, ispec_selected_source, ibool);
    transfer_field_from_device(mp,mp->d_veloc,veloc, number_receiver_global,
             mp->d_ispec_selected_source, ispec_selected_source, ibool);
    transfer_field_from_device(mp,mp->d_accel,accel, number_receiver_global,
             mp->d_ispec_selected_source, ispec_selected_source, ibool);
  }
  else if(SIMULATION_TYPE == 3) {
    transfer_field_from_device(mp,mp->d_b_displ,b_displ, number_receiver_global,
             mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_from_device(mp,mp->d_b_veloc,b_veloc, number_receiver_global,
             mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_from_device(mp,mp->d_b_accel,b_accel, number_receiver_global,
             mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
  }

}

/* ----------------------------------------------------------------------------------------------- */

// ACOUSTIC simulations

/* ----------------------------------------------------------------------------------------------- */

__global__ void transfer_stations_fields_acoustic_from_device_kernel(int* number_receiver_global,
                                                                     int* ispec_selected_rec,
                                                                     int* ibool,
                                                                     float* station_seismo_potential,
                                                                     float* desired_potential) {

  int blockID = blockIdx.x + blockIdx.y*gridDim.x;
  int nodeID = threadIdx.x + blockID*blockDim.x;

  int irec = number_receiver_global[blockID]-1;
  int ispec = ispec_selected_rec[irec]-1;
  int iglob = ibool[threadIdx.x + 125*ispec]-1;

  //if(threadIdx.x == 0 ) printf("node acoustic: %i %i %i %i %i %e \n",blockID,nodeID,irec,ispec,iglob,desired_potential[iglob]);

  station_seismo_potential[nodeID] = desired_potential[iglob];
}

/* ----------------------------------------------------------------------------------------------- */

void transfer_field_acoustic_from_device(Mesh* mp,
                                         float* d_potential,
                                         float* h_potential,
                                         int* number_receiver_global,
                                         int* d_ispec_selected,
                                         int* h_ispec_selected,
                                         int* ibool) {

TRACE("transfer_field_acoustic_from_device");

  int irec_local,irec,ispec,iglob,j;

  // checks if anything to do
  if( mp->nrec_local == 0 ) return;

  // sets up kernel dimensions
  int blocksize = 125;
  int num_blocks_x = mp->nrec_local;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(blocksize,1,1);

  // prepare field transfer array on device
  transfer_stations_fields_acoustic_from_device_kernel<<<grid,threads>>>(mp->d_number_receiver_global,
                                                                         d_ispec_selected,
                                                                         mp->d_ibool,
                                                                         mp->d_station_seismo_potential,
                                                                         d_potential);


  print_CUDA_error_if_any(cudaMemcpy(mp->h_station_seismo_potential,mp->d_station_seismo_potential,
                                     mp->nrec_local*125*sizeof(float),cudaMemcpyDeviceToHost),500);

  //printf("copy local receivers: %i \n",mp->nrec_local);

  for(irec_local=0; irec_local < mp->nrec_local; irec_local++) {
    irec = number_receiver_global[irec_local]-1;
    ispec = h_ispec_selected[irec]-1;

    // copy element values
    // note: iglob may vary and can be irregularly accessing the h_potential array
    for(j=0; j < 125; j++){
      iglob = ibool[j+125*ispec]-1;
      h_potential[iglob] = mp->h_station_seismo_potential[j+irec_local*125];
    }

    // copy each station element's points to working array
    // note: this works if iglob values would be all aligned...
    //memcpy(&(h_potential[iglob]),&(mp->h_station_seismo_potential[irec_local*125]),125*sizeof(float));

  }
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("transfer_field_acoustic_from_device");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

extern "C"
void FC_FUNC_(transfer_station_ac_from_device,
              TRANSFER_STATION_AC_FROM_DEVICE)(
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
                                                int* SIMULATION_TYPEf) {

TRACE("transfer_station_ac_from_device");
  //double start_time = get_time();

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper
  // checks if anything to do
  if( mp->nrec_local == 0 ) return;

  int SIMULATION_TYPE = *SIMULATION_TYPEf;

  if(SIMULATION_TYPE == 1) {
    transfer_field_acoustic_from_device(mp,mp->d_potential_acoustic,potential_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_acoustic_from_device(mp,mp->d_potential_dot_acoustic,potential_dot_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_acoustic_from_device(mp,mp->d_potential_dot_dot_acoustic,potential_dot_dot_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
  }
  else if(SIMULATION_TYPE == 2) {
    transfer_field_acoustic_from_device(mp,mp->d_potential_acoustic,potential_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_source, ispec_selected_source, ibool);
    transfer_field_acoustic_from_device(mp,mp->d_potential_dot_acoustic,potential_dot_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_source, ispec_selected_source, ibool);
    transfer_field_acoustic_from_device(mp,mp->d_potential_dot_dot_acoustic,potential_dot_dot_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_source, ispec_selected_source, ibool);
  }
  else if(SIMULATION_TYPE == 3) {
    transfer_field_acoustic_from_device(mp,mp->d_b_potential_acoustic,b_potential_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_acoustic_from_device(mp,mp->d_b_potential_dot_acoustic,b_potential_dot_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
    transfer_field_acoustic_from_device(mp,mp->d_b_potential_dot_dot_acoustic,b_potential_dot_dot_acoustic,
                                        number_receiver_global,
                                        mp->d_ispec_selected_rec, ispec_selected_rec, ibool);
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  //double end_time = get_time();
  //printf("Elapsed time: %e\n",end_time-start_time);
  exit_on_cuda_error("transfer_station_ac_from_device");
#endif
}

