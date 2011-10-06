#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "config.h"
#include "mesh_constants_cuda.h"
// #include "epik_user.h"


/* ----------------------------------------------------------------------------------------------- */

// elastic domain sources

/* ----------------------------------------------------------------------------------------------- */


// crashes if the CMTSOLUTION does not match the mesh properly
__global__ void compute_add_sources_kernel(float* accel, int* ibool, int* ispec_is_inner, int phase_is_inner, float* sourcearrays, double* stf_pre_compute,int myrank, int* islice_selected_source, int* ispec_selected_source, int* ispec_is_elastic, int NSOURCES,float* d_debug) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;
  
  int isource  = blockIdx.x + gridDim.x*blockIdx.y; // bx
  int ispec;
  int iglob;
  double stf;

  if(isource < NSOURCES) { // when NSOURCES > 65535, but mod(nspec_top,2) > 0, we end up with an extra block.
    
    if(myrank == islice_selected_source[isource]) {

      ispec = ispec_selected_source[isource]-1;

      if(ispec_is_inner[ispec] == phase_is_inner && ispec_is_elastic[ispec] == 1) {
		    
        stf = stf_pre_compute[isource];
        iglob = ibool[INDEX4(5,5,5,i,j,k,ispec)]-1;
        atomicAdd(&accel[iglob*3],
                  sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 0, i,j,k)]*stf);
        atomicAdd(&accel[iglob*3+1],
                  sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 1, i,j,k)]*stf);
	// if((iglob*3+2 == 304598)) {
	//   atomicAdd(&d_debug[0],1.0f);
	//   d_debug[1] = accel[iglob*3+2];
	//   d_debug[2] = sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 2, i,j,k)];
	//   d_debug[3] = stf;
	// }
	// d_debug[4] = 42.0f;
        atomicAdd(&accel[iglob*3+2],
                  sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 2, i,j,k)]*stf);	
      }
    }
  }
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(add_sourcearrays_adjoint_cuda,
              ADD_SOURCEARRAYS_ADJOINT_CUDA)(long* Mesh_pointer,
                                              int* USE_FORCE_POINT_SOURCE,
                                              double* h_stf_pre_compute,int* NSOURCES,
                                              int* phase_is_inner,int* myrank) {
TRACE("add_sourcearrays_adjoint_cuda");
// EPIK_TRACER("add_sourcearrays_adjoint_cuda");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  if(*USE_FORCE_POINT_SOURCE) {
    printf("USE FORCE POINT SOURCE not implemented for GPU_MODE");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  print_CUDA_error_if_any(cudaMemcpy(mp->d_stf_pre_compute,h_stf_pre_compute,(*NSOURCES)*sizeof(double),cudaMemcpyHostToDevice),18);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING  
  exit_on_cuda_error("noise_read_add_surface_movie_cuda_kernel");
#endif
  
  int num_blocks_x = *NSOURCES;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(5,5,5);

  float* d_debug;
  // float* h_debug = (float*)calloc(128,sizeof(float));
  // cudaMalloc((void**)&d_debug,128*sizeof(float));
  // cudaMemcpy(d_debug,h_debug,128*sizeof(float),cudaMemcpyHostToDevice);
  
  compute_add_sources_kernel<<<grid,threads>>>(mp->d_b_accel,mp->d_ibool, 
                                               mp->d_ispec_is_inner, *phase_is_inner, 
                                               mp->d_sourcearrays, 
                                               mp->d_stf_pre_compute,
                                               *myrank, 
                                               mp->d_islice_selected_source,mp->d_ispec_selected_source,
                                               mp->d_ispec_is_elastic, 
                                               *NSOURCES,
                                               d_debug);

  // cudaMemcpy(h_debug,d_debug,128*sizeof(float),cudaMemcpyDeviceToHost);
  // for(int i=0;i<10;i++) {
  //   printf("debug[%d] = %e \n",i,h_debug[i]);
  // }
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("add_sourcearrays_adjoint_cuda");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(compute_add_sources_elastic_cuda,
              COMPUTE_ADD_SOURCES_ELASTIC_CUDA)(long* Mesh_pointer_f, 
                                                int* NSPEC_ABf, int* NGLOB_ABf, 
                                                int* phase_is_innerf,int* NSOURCESf, 
                                                int* itf, float* dtf, float* t0f,
                                                int* SIMULATION_TYPEf,int* NSTEPf,
                                                int* NOISE_TOMOGRAPHYf, 
                                                int* USE_FORCE_POINT_SOURCEf, 
                                                double* h_stf_pre_compute, int* myrankf) {

TRACE("compute_add_sources_elastic_cuda");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  //int NSPEC_AB = *NSPEC_ABf;
  //int NGLOB_AB = *NGLOB_ABf;
  int phase_is_inner = *phase_is_innerf;
  //int it = *itf;
  //float dt = *dtf;
  //float t0 = *t0f;
  //int SIMULATION_TYPE = *SIMULATION_TYPEf;
  //int NSTEP = *NSTEPf;
  //int NOISE_TOMOGRAPHY = *NOISE_TOMOGRAPHYf;
  int NSOURCES = *NSOURCESf;
  //int USE_FORCE_POINT_SOURCE = *USE_FORCE_POINT_SOURCEf;
  int myrank = *myrankf;

  float* d_debug;
  int num_blocks_x = NSOURCES;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  //double* d_stf_pre_compute;
  print_CUDA_error_if_any(cudaMemcpy(mp->d_stf_pre_compute,h_stf_pre_compute,NSOURCES*sizeof(double),cudaMemcpyHostToDevice),18);
  
  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(5,5,5);
  // (float* accel, int* ibool, int* ispec_is_inner, int phase_is_inner, float* sourcearrays, double* stf_pre_compute,int myrank, int* islice_selected_source, int* ispec_selected_source, int* ispec_is_elastic, int NSOURCES)
  
  
  
  compute_add_sources_kernel<<<grid,threads>>>(mp->d_accel,mp->d_ibool, mp->d_ispec_is_inner, phase_is_inner, mp->d_sourcearrays, mp->d_stf_pre_compute,myrank, mp->d_islice_selected_source,mp->d_ispec_selected_source,mp->d_ispec_is_elastic, NSOURCES,d_debug);

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("compute_add_sources_kernel");
#endif
}

/* ----------------------------------------------------------------------------------------------- */

__global__ void add_source_master_rec_noise_cuda_kernel(int* ibool, int* ispec_selected_rec, int irec_master_noise, real* accel, real* noise_sourcearray, int it) {
  int tx = threadIdx.x;
  int iglob = ibool[tx + 125*(ispec_selected_rec[irec_master_noise-1]-1)]-1;

  // not sure if we need atomic operations but just in case...
  // accel[3*iglob] += noise_sourcearray[3*tx + 3*125*it];
  // accel[1+3*iglob] += noise_sourcearray[1+3*tx + 3*125*it];
  // accel[2+3*iglob] += noise_sourcearray[2+3*tx + 3*125*it];
  
  atomicAdd(&accel[iglob*3],noise_sourcearray[3*tx + 3*125*it]);
  atomicAdd(&accel[iglob*3+1],noise_sourcearray[1+3*tx + 3*125*it]);
  atomicAdd(&accel[iglob*3+2],noise_sourcearray[2+3*tx + 3*125*it]);
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(add_source_master_rec_noise_cuda,
              ADD_SOURCE_MASTER_REC_NOISE_CUDA)(long* Mesh_pointer_f, 
                                                int* myrank_f,  
                                                int* it_f, 
                                                int* irec_master_noise_f, 
                                                int* islice_selected_rec) {

TRACE("add_source_master_rec_noise_cuda");

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container

  int it = *it_f-1; // -1 for Fortran -> C indexing differences
  int irec_master_noise = *irec_master_noise_f;
  int myrank = *myrank_f;
  dim3 grid(1,1,1);
  dim3 threads(125,1,1);
  if(myrank == islice_selected_rec[irec_master_noise-1]) {
    add_source_master_rec_noise_cuda_kernel<<<grid,threads>>>(mp->d_ibool, mp->d_ispec_selected_rec,
							      irec_master_noise, mp->d_accel,
							      mp->d_noise_sourcearray, it);    

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("add_source_master_rec_noise_cuda_kernel");
#endif
  }
}

/* ----------------------------------------------------------------------------------------------- */

__global__ void add_sources_SIM_TYPE_2_OR_3_kernel(float* accel, int nrec,		  
						   float* adj_sourcearrays,
						   int* ibool,
						   int* ispec_is_inner,
						   int* ispec_selected_rec,
						   int phase_is_inner,
						   int* islice_selected_rec,
						   int* pre_computed_irec,
						   int nadj_rec_local,
						   int NTSTEP_BETWEEN_ADJSRC,
						   int myrank,
						   int* debugi,
						   float* debugf) {
  int irec_local = blockIdx.x + gridDim.x*blockIdx.y;
  if(irec_local<nadj_rec_local) { // when nrec > 65535, but mod(nspec_top,2) > 0, we end up with an extra block.

    int irec = pre_computed_irec[irec_local];
    
    int ispec_selected = ispec_selected_rec[irec]-1;
    if(ispec_is_inner[ispec_selected] == phase_is_inner) {
      int i = threadIdx.x;
      int j = threadIdx.y;
      int k = threadIdx.z;
      int iglob = ibool[i+5*(j+5*(k+5*ispec_selected))]-1;
      
      // atomic operations are absolutely necessary for correctness!
      atomicAdd(&(accel[0+3*iglob]),adj_sourcearrays[INDEX5(5,5,5,3,
							    i,j,k,
							    0,
							    irec_local)]);
		
      atomicAdd(&accel[1+3*iglob], adj_sourcearrays[INDEX5(5,5,5,3,
							   i,j,k,
							   1,
							   irec_local)]);
      
      atomicAdd(&accel[2+3*iglob],adj_sourcearrays[INDEX5(5,5,5,3,
							  i,j,k,
							  2,
							  irec_local)]);
    }
     
  }
  
}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(add_sources_sim_type_2_or_3,
              ADD_SOURCES_SIM_TYPE_2_OR_3)(long* Mesh_pointer, 
                                           float* h_adj_sourcearrays,
                                           int* size_adj_sourcearrays, int* ispec_is_inner,
                                           int* phase_is_inner, int* ispec_selected_rec,
                                           int* ibool,
                                           int* myrank, int* nrec, int* time_index,
                                           int* h_islice_selected_rec,int* nadj_rec_local,
                                           int* NTSTEP_BETWEEN_READ_ADJSRC) {

TRACE("add_sources_sim_type_2_or_3");

  if(*nadj_rec_local > 0) {
  
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
    // make sure grid dimension is less than 65535 in x dimension
    int num_blocks_x = *nadj_rec_local;
    int num_blocks_y = 1;
    while(num_blocks_x > 65535) {
      num_blocks_x = ceil(num_blocks_x/2.0);
      num_blocks_y = num_blocks_y*2;
    }
    dim3 grid(num_blocks_x,num_blocks_y,1);
    dim3 threads(5,5,5);
  
    float* d_adj_sourcearrays;
    print_CUDA_error_if_any(cudaMalloc((void**)&d_adj_sourcearrays,
				       (*nadj_rec_local)*3*125*sizeof(float)),1);
    float* h_adj_sourcearrays_slice = (float*)malloc((*nadj_rec_local)*3*125*sizeof(float));

    int* h_pre_computed_irec = new int[*nadj_rec_local];
    int* d_pre_computed_irec;
    cudaMalloc((void**)&d_pre_computed_irec,(*nadj_rec_local)*sizeof(int));
  
    // build slice of adj_sourcearrays because full array is *very* large.
    int irec_local = 0;
    for(int irec = 0;irec<*nrec;irec++) {
      if(*myrank == h_islice_selected_rec[irec]) {
	irec_local++;
	h_pre_computed_irec[irec_local-1] = irec;
	if(ispec_is_inner[ispec_selected_rec[irec]-1] == *phase_is_inner) {
	  for(int k=0;k<5;k++) {
	    for(int j=0;j<5;j++) {
	      for(int i=0;i<5;i++) {

		h_adj_sourcearrays_slice[INDEX5(5,5,5,3,
						i,j,k,0,
						irec_local-1)]
		  = h_adj_sourcearrays[INDEX6(*nadj_rec_local,
					      *NTSTEP_BETWEEN_READ_ADJSRC,
					      3,5,5,
					      irec_local-1,
					      *time_index-1,
					      0,i,j,k)];
	      
		h_adj_sourcearrays_slice[INDEX5(5,5,5,3,
						i,j,k,1,
						irec_local-1)]
		  = h_adj_sourcearrays[INDEX6(*nadj_rec_local,
					      *NTSTEP_BETWEEN_READ_ADJSRC,
					      3,5,5,
					      irec_local-1,
					      *time_index-1,
					      1,i,j,k)];
	      
		h_adj_sourcearrays_slice[INDEX5(5,5,5,3,
						i,j,k,2,
						irec_local-1)]
		  = h_adj_sourcearrays[INDEX6(*nadj_rec_local,
					      *NTSTEP_BETWEEN_READ_ADJSRC,
					      3,5,5,
					      irec_local-1,
					      *time_index-1,
					      2,i,j,k)];
	      
								   
	      }
	    }
	  }
	}
      }
    }
    // printf("irec_local vs. *nadj_rec_local -> %d vs. %d\n",irec_local,*nadj_rec_local);
    // for(int ispec=0;ispec<(*nadj_rec_local);ispec++) {
    //   for(int i=0;i<5;i++)
    //     for(int j=0;j<5;j++)
    // 	for(int k=0;k<5;k++) {
    // 	  h_adj_sourcearrays_slice[INDEX5(5,5,5,3,i,j,k,0,ispec)] =
    // 	    h_adj_sourcearrays[INDEX6(*nadj_rec_local,*NTSTEP_BETWEEN_READ_ADJSRC,3,5,5,
    // 				      ispec,
    // 				      *time_index-1,
    // 				      0,
    // 				      i,j,k)];
    // 	  h_adj_sourcearrays_slice[INDEX5(5,5,5,3,i,j,k,1,ispec)] =
    // 	    h_adj_sourcearrays[INDEX6(*nadj_rec_local,*NTSTEP_BETWEEN_READ_ADJSRC,3,5,5,
    // 				      ispec,
    // 				      *time_index-1,
    // 				      1,
    // 				      i,j,k)];
    // 	  h_adj_sourcearrays_slice[INDEX5(5,5,5,3,i,j,k,2,ispec)] =
    // 	    h_adj_sourcearrays[INDEX6(*nadj_rec_local,*NTSTEP_BETWEEN_ADJSRC,3,5,5,
    // 				      ispec,
    // 				      *time_index-1,
    // 				      2,
    // 				      i,j,k)];	  
    // 	}
    
    // }
  
    cudaMemcpy(d_adj_sourcearrays, h_adj_sourcearrays_slice,(*nadj_rec_local)*3*125*sizeof(float),
	       cudaMemcpyHostToDevice);
  

    // the irec_local variable needs to be precomputed (as
    // h_pre_comp..), because normally it is in the loop updating accel,
    // and due to how it's incremented, it cannot be parallelized
  
    // int irec_local=0;
    // for(int irec=0;irec<*nrec;irec++) {
    //   if(*myrank == h_islice_selected_rec[irec]) {
    //     h_pre_computed_irec_local_index[irec] = irec_local;
    //     irec_local++;
    //     if(irec_local==1) {
    // 	// printf("%d:first useful irec==%d\n",rank,irec);
    //     }
    //   }
    //   else h_pre_computed_irec_local_index[irec] = 0;
    // }
    cudaMemcpy(d_pre_computed_irec,h_pre_computed_irec,
	       (*nadj_rec_local)*sizeof(int),cudaMemcpyHostToDevice);
    // pause_for_debugger(1);
    int* d_debugi, *h_debugi;
    float* d_debugf, *h_debugf;
    h_debugi = (int*)calloc(num_blocks_x,sizeof(int));
    cudaMalloc((void**)&d_debugi,num_blocks_x*sizeof(int));
    cudaMemcpy(d_debugi,h_debugi,num_blocks_x*sizeof(int),cudaMemcpyHostToDevice);
    h_debugf = (float*)calloc(num_blocks_x,sizeof(float));
    cudaMalloc((void**)&d_debugf,num_blocks_x*sizeof(float));
    cudaMemcpy(d_debugf,h_debugf,num_blocks_x*sizeof(float),cudaMemcpyHostToDevice);
      
    add_sources_SIM_TYPE_2_OR_3_kernel<<<grid,threads>>>(mp->d_accel, *nrec,
							 d_adj_sourcearrays, mp->d_ibool,
							 mp->d_ispec_is_inner,
							 mp->d_ispec_selected_rec,
							 *phase_is_inner,
							 mp->d_islice_selected_rec,
							 d_pre_computed_irec,
							 *nadj_rec_local,
							 *NTSTEP_BETWEEN_READ_ADJSRC,
							 *myrank,
							 d_debugi,d_debugf);

    cudaMemcpy(h_debugi,d_debugi,num_blocks_x*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_debugf,d_debugf,num_blocks_x*sizeof(float),cudaMemcpyDeviceToHost);
  
    // printf("%d: pre_com0:%d\n",rank,h_pre_computed_irec_local_index[0]);
    // printf("%d: pre_com1:%d\n",rank,h_pre_computed_irec_local_index[1]);
    // printf("%d: pre_com2:%d\n",rank,h_pre_computed_irec_local_index[2]);
    // for(int i=156;i<(156+30);i++) {
    //   if(rank==0) printf("%d:debug[%d] = i/f = %d / %e\n",rank,i,h_debugi[i],h_debugf[i]);
    // }
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
    // MPI_Barrier(MPI_COMM_WORLD);
    exit_on_cuda_error("add_sources_SIM_TYPE_2_OR_3_kernel");    
  
    // printf("Proc %d exiting with successful kernel\n",rank);
    // exit(1);
#endif
    delete h_pre_computed_irec;
    cudaFree(d_adj_sourcearrays);
    cudaFree(d_pre_computed_irec);
  }
}

/* ----------------------------------------------------------------------------------------------- */

// acoustic sources

/* ----------------------------------------------------------------------------------------------- */

__global__ void compute_add_sources_acoustic_kernel(float* potential_dot_dot_acoustic, 
                                                    int* ibool, 
                                                    int* ispec_is_inner, 
                                                    int phase_is_inner, 
                                                    float* sourcearrays, 
                                                    double* stf_pre_compute,
                                                    int myrank, 
                                                    int* islice_selected_source, 
                                                    int* ispec_selected_source, 
                                                    int* ispec_is_acoustic, 
                                                    float* kappastore,
                                                    int NSOURCES) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;
  
  int isource  = blockIdx.x + gridDim.x*blockIdx.y; // bx
  
  int ispec;
  int iglob;
  double stf;
  float kappal;
  
  if( isource < NSOURCES ){
    
    //if(myrank == 0 && i== 0 && j == 0 && k == 0) printf("source isource = %i \n",isource);
    
    if(myrank == islice_selected_source[isource]) {
      
      ispec = ispec_selected_source[isource]-1;
      
      if(ispec_is_inner[ispec] == phase_is_inner && ispec_is_acoustic[ispec] == 1) {
        
        stf = stf_pre_compute[isource];
        iglob = ibool[INDEX4(5,5,5,i,j,k,ispec)]-1;
        kappal = kappastore[INDEX4(5,5,5,i,j,k,ispec)];
        
        //printf("source ispec = %i %i %e %e \n",ispec,iglob,stf,kappal);
        //printf("source arr = %e %i %i %i %i %i\n", -sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 0, i,j,k)]*stf/kappal,i,j,k,iglob,ispec);
        
        atomicAdd(&potential_dot_dot_acoustic[iglob],
                  -sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 0, i,j,k)]*stf/kappal);
        
        //      potential_dot_dot_acoustic[iglob] +=
        //                -sourcearrays[INDEX5(NSOURCES, 3, 5, 5,isource, 0, i,j,k)]*stf/kappal; 
        
        //printf("potential = %e %i %i %i %i %i\n", potential_dot_dot_acoustic[iglob],i,j,k,iglob,ispec);
        
        
      }
    }
  }  
}


/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(compute_add_sources_acoustic_cuda,
              COMPUTE_ADD_SOURCES_ACOUSTIC_CUDA)(long* Mesh_pointer_f, 
                                                 int* phase_is_innerf,
                                                 int* NSOURCESf, 
                                                 int* SIMULATION_TYPEf,
                                                 int* USE_FORCE_POINT_SOURCEf, 
                                                 double* h_stf_pre_compute, 
                                                 int* myrankf) {
  
TRACE("compute_add_sources_acoustic_cuda");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  int phase_is_inner = *phase_is_innerf;
  //int SIMULATION_TYPE = *SIMULATION_TYPEf;
  int NSOURCES = *NSOURCESf;
  //int USE_FORCE_POINT_SOURCE = *USE_FORCE_POINT_SOURCEf;
  int myrank = *myrankf;  
  
  int num_blocks_x = NSOURCES;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  // copies pre-computed source time factors onto GPU  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_stf_pre_compute,h_stf_pre_compute,NSOURCES*sizeof(double),cudaMemcpyHostToDevice),18);
  
  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(5,5,5);
  
  compute_add_sources_acoustic_kernel<<<grid,threads>>>(mp->d_potential_dot_dot_acoustic,
                                                        mp->d_ibool, 
                                                        mp->d_ispec_is_inner, 
                                                        phase_is_inner, 
                                                        mp->d_sourcearrays, 
                                                        mp->d_stf_pre_compute,
                                                        myrank, 
                                                        mp->d_islice_selected_source,
                                                        mp->d_ispec_selected_source,
                                                        mp->d_ispec_is_acoustic, 
                                                        mp->d_kappastore,
                                                        NSOURCES);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("compute_add_sources_acoustic_cuda");  
#endif    
}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(compute_add_sources_acoustic_sim3_cuda,
              COMPUTE_ADD_SOURCES_ACOUSTIC_SIM3_CUDA)(long* Mesh_pointer_f, 
                                                      int* phase_is_innerf,
                                                      int* NSOURCESf, 
                                                      int* SIMULATION_TYPEf,
                                                      int* USE_FORCE_POINT_SOURCEf, 
                                                      double* h_stf_pre_compute, 
                                                      int* myrankf) {
  
TRACE("compute_add_sources_acoustic_sim3_cuda");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  int phase_is_inner = *phase_is_innerf;
  //int SIMULATION_TYPE = *SIMULATION_TYPEf;
  int NSOURCES = *NSOURCESf;
  //int USE_FORCE_POINT_SOURCE = *USE_FORCE_POINT_SOURCEf;
  int myrank = *myrankf;  
  
  int num_blocks_x = NSOURCES;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  // copies source time factors onto GPU  
  print_CUDA_error_if_any(cudaMemcpy(mp->d_stf_pre_compute,h_stf_pre_compute,NSOURCES*sizeof(double),cudaMemcpyHostToDevice),18);
  
  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(5,5,5);
  
  compute_add_sources_acoustic_kernel<<<grid,threads>>>(mp->d_b_potential_dot_dot_acoustic,
                                                        mp->d_ibool, 
                                                        mp->d_ispec_is_inner, 
                                                        phase_is_inner, 
                                                        mp->d_sourcearrays, 
                                                        mp->d_stf_pre_compute,
                                                        myrank, 
                                                        mp->d_islice_selected_source,
                                                        mp->d_ispec_selected_source,
                                                        mp->d_ispec_is_acoustic, 
                                                        mp->d_kappastore,
                                                        NSOURCES);
    
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("compute_add_sources_acoustic_sim3_cuda");  
#endif    
}


/* ----------------------------------------------------------------------------------------------- */

// acoustic adjoint sources

/* ----------------------------------------------------------------------------------------------- */

__global__ void add_sources_acoustic_SIM_TYPE_2_OR_3_kernel(float* potential_dot_dot_acoustic, 
                                                            int nrec,
                                                            int pre_computed_index,
                                                            float* adj_sourcearrays,
                                                            int* ibool,
                                                            int* ispec_is_inner,
                                                            int* ispec_is_acoustic,
                                                            float* kappastore,
                                                            int* ispec_selected_rec,
                                                            int phase_is_inner,
                                                            int* islice_selected_rec,
                                                            int* pre_computed_irec_local_index,
                                                            int nadj_rec_local,
                                                            int NTSTEP_BETWEEN_ADJSRC,
                                                            int myrank) {
  
  int irec = blockIdx.x + gridDim.x*blockIdx.y;
  
  //float kappal;
  int i,j,k,iglob,ispec;
  
  // because of grid shape, irec can be too big
  if(irec<nrec) { 
    
    // adds source only if this proc carries the sources
    if( myrank == islice_selected_rec[irec] ){
      
      // adds acoustic source
      ispec = ispec_selected_rec[irec]-1;      
      if( ispec_is_acoustic[ispec] == 1 ){
        
        // checks if element is in phase_is_inner run
        if(ispec_is_inner[ispec] == phase_is_inner) {
          i = threadIdx.x;
          j = threadIdx.y;
          k = threadIdx.z;
          iglob = ibool[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)]-1;
          
          //kappal = kappastore[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)];
          
          //potential_dot_dot_acoustic[iglob] += adj_sourcearrays[INDEX6(nadj_rec_local,NTSTEP_BETWEEN_ADJSRC,3,5,5,
          //                                            pre_computed_irec_local_index[irec],
          //                                            pre_computed_index,
          //                                            0,
          //                                            i,j,k)]/kappal;            

          // beware, for acoustic medium, a pressure source would be taking the negative 
          // and divide by Kappa of the fluid;
          // this would have to be done when constructing the adjoint source.
          //
          // note: we take the first component of the adj_sourcearrays
          //          the idea is to have e.g. a pressure source, where all 3 components would be the same
          
          atomicAdd(&potential_dot_dot_acoustic[iglob],
                    +adj_sourcearrays[INDEX6(nadj_rec_local,NTSTEP_BETWEEN_ADJSRC,3,5,5,
                                             pre_computed_irec_local_index[irec],pre_computed_index-1,
                                             0,i,j,k)] // / kappal 
                                             );
        }    
      }  
    }
  }
}

/* ----------------------------------------------------------------------------------------------- */


extern "C" 
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
                                                         int* NTSTEP_BETWEEN_ADJSRC) {
  
TRACE("add_sources_acoustic_sim_type_2_or_3_cuda");
  
  Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
  
  // make sure grid dimension is less than 65535 in x dimension
  int num_blocks_x = *nrec;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }
  dim3 grid(num_blocks_x,num_blocks_y,1);
  dim3 threads(5,5,5);
  
  // copies source arrays
  // daniel: todo workaround -- adj_sourcearrays can be very big, but here only at 
  //             specific time it (pre_computed_irec_local_index) is actually needed...
  float* d_adj_sourcearrays;
  print_CUDA_error_if_any(cudaMalloc((void**)&d_adj_sourcearrays,
                                     (*size_adj_sourcearrays)*sizeof(float)),731);
  print_CUDA_error_if_any(cudaMemcpy(d_adj_sourcearrays, h_adj_sourcearrays,
                                     (*size_adj_sourcearrays)*sizeof(float),cudaMemcpyHostToDevice),732);
  
  //int* h_pre_computed_irec_local_index = new int[*nadj_rec_local];
  int* h_pre_computed_irec_local_index = (int*) calloc(*nrec,sizeof(int));
  
  int* d_pre_computed_irec_local_index;
  print_CUDA_error_if_any(cudaMalloc((void**)&d_pre_computed_irec_local_index,
                                     (*nrec)*sizeof(int)),741);
  
  // the irec_local variable needs to be precomputed (as
  // h_pre_comp..), because normally it is in the loop updating accel,
  // and due to how it's incremented, it cannot be parallized
  int irec_local=0;
  for(int irec=0;irec<*nrec;irec++) {
    if(*myrank == h_islice_selected_rec[irec]) {
      h_pre_computed_irec_local_index[irec] = irec_local;
      irec_local++;      
    }
    else h_pre_computed_irec_local_index[irec] = 0;
  }
  
  //daniel
  //printf("irec local: rank=%d irec_local=%d nadj_rec_local=%d nrec=%d \n",*myrank,irec_local,*nadj_rec_local,*nrec);
  
  print_CUDA_error_if_any(cudaMemcpy(d_pre_computed_irec_local_index,h_pre_computed_irec_local_index,
                                     (*nrec)*sizeof(int),cudaMemcpyHostToDevice),742);
  
  add_sources_acoustic_SIM_TYPE_2_OR_3_kernel<<<grid,threads>>>(mp->d_potential_dot_dot_acoustic, 
                                                                *nrec, 
                                                                *pre_computed_index,
                                                                d_adj_sourcearrays, 
                                                                mp->d_ibool,
                                                                mp->d_ispec_is_inner,
                                                                mp->d_ispec_is_acoustic,
                                                                mp->d_kappastore,
                                                                mp->d_ispec_selected_rec,
                                                                *phase_is_inner,                                                       
                                                                mp->d_islice_selected_rec,
                                                                d_pre_computed_irec_local_index,
                                                                *nadj_rec_local,
                                                                *NTSTEP_BETWEEN_ADJSRC,
                                                                *myrank);
  
  //delete h_pre_computed_irec_local_index;
  free(h_pre_computed_irec_local_index);
  cudaFree(d_adj_sourcearrays);
  cudaFree(d_pre_computed_irec_local_index);  
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("add_sources_acoustic_SIM_TYPE_2_OR_3_kernel");  
#endif    
}
