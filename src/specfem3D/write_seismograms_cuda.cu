#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>

#include "mesh_constants_cuda.h"

#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + (y)*xsize + (z)*xsize*ysize
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize
#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize + (j)*xsize*ysize*zsize*isize


#define ENABLE_VERY_SLOW_ERROR_CHECKING


__global__ void transfer_stations_fields_from_device_kernel(int* number_receiver_global,
							    int* ispec_selected_rec,
							    int* ibool,
							    float* station_seismo_field,
							    float* desired_field,
							    int nrec_local,int* debug_index) {
  int blockID = blockIdx.x + blockIdx.y*gridDim.x;
  if(blockID<nrec_local) {
    int nodeID = threadIdx.x + blockID*blockDim.x;
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

extern "C" void pause_for_debuger(int);

void transfer_field_from_device(Mesh* mp, float* d_field,float* h_field,
				int* number_receiver_global,
				int* d_ispec_selected,
				int* h_ispec_selected,
				int* ibool) {
  
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

  int* d_debug_index,*h_debug_index;
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
  
  // // sync and check to catch errors from previous async operations
  // cudaThreadSynchronize();
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess)
  //   {
  //     fprintf(stderr,"Error launching/running transfer_stations_fields_from_device_kernel: %s\n", cudaGetErrorString(err));
  //     exit(1);
  //   }
#endif
  
  cudaMemcpy(mp->h_station_seismo_field,mp->d_station_seismo_field,
	     (3*125)*(mp->nrec_local)*sizeof(float),cudaMemcpyDeviceToHost);
    
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  // sync and check to catch errors from previous async operations
  exit_on_cuda_error("transfer_stations_fields_from_device_kernel_memcpy");
  // cudaThreadSynchronize();
  // err = cudaGetLastError();
  // if (err != cudaSuccess)
  // {
  // fprintf(stderr,"Error launching/running transfer_stations_fields_from_device_kernel_memcpy: %s\n", cudaGetErrorString(err));
  // exit(1);
  // }
#endif
  
  // pause_for_debug(1);
  int irec_local;
  
  for(irec_local=0;irec_local<mp->nrec_local;irec_local++) {
    int irec = number_receiver_global[irec_local]-1;
    int ispec = h_ispec_selected[irec]-1;
       
    for(int i=0;i<125;i++) {
      int iglob = ibool[i+125*ispec]-1;
      h_field[0+3*iglob] = mp->h_station_seismo_field[0+3*i+irec_local*125*3];
      h_field[1+3*iglob] = mp->h_station_seismo_field[1+3*i+irec_local*125*3];
      h_field[2+3*iglob] = mp->h_station_seismo_field[2+3*i+irec_local*125*3];      
    }

  }
}

extern "C" void transfer_station_fields_from_device_(float* displ,float* veloc,float* accel,
						     float* b_displ, float* b_veloc, float* b_accel,
						     long* Mesh_pointer_f,int* number_receiver_global,
						     int* ispec_selected_rec,int* ispec_selected_source,
						     int* ibool,int* SIMULATION_TYPEf) {

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper
   

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

// extern "C" void save_seismogram(char* 
