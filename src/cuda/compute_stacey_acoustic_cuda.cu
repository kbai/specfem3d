#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "config.h"
#include "mesh_constants_cuda.h"


/* ----------------------------------------------------------------------------------------------- */

__global__ void compute_stacey_acoustic_kernel(float* potential_dot_acoustic, 
                                               float* potential_dot_dot_acoustic, 
                                               int* abs_boundary_ispec,
                                               int* abs_boundary_ijk, 
                                               int* ibool,
                                               float* rhostore,
                                               float* kappastore,
                                               real* abs_boundary_jacobian2Dw,
                                               int* ispec_is_inner, 
                                               int* ispec_is_acoustic,
                                               int phase_is_inner,
                                               int SIMULATION_TYPE, int SAVE_FORWARD,
                                               float* b_potential_dot_acoustic,
                                               float* b_potential_dot_dot_acoustic,
                                               float* b_absorb_potential // ,float* debug_val,int* debug_val_int
                                               ) {

  int igll = threadIdx.x; // tx
  int iface = blockIdx.x + gridDim.x*blockIdx.y; // bx

  
  int i,j,k,iglob,ispec;
  realw rhol,kappal,cpl;
  realw jacobianw;
  
  // don't compute points outside NGLLSQUARE==NGLL2==25  
  // way 2: no further check needed since blocksize = 25
  //  if(igll<NGLL2) {    
    
    // "-1" from index values to convert from Fortran-> C indexing
    ispec = abs_boundary_ispec[iface]-1;
    
    if(ispec_is_inner[ispec] == phase_is_inner && ispec_is_acoustic[ispec]==1) {

      i = abs_boundary_ijk[INDEX3(NDIM,NGLL2,0,igll,iface)]-1;
      j = abs_boundary_ijk[INDEX3(NDIM,NGLL2,1,igll,iface)]-1;
      k = abs_boundary_ijk[INDEX3(NDIM,NGLL2,2,igll,iface)]-1;
      iglob = ibool[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)]-1;
      
      // determines bulk sound speed
      rhol = rhostore[INDEX4_PADDED(NGLLX,NGLLX,NGLLX,i,j,k,ispec)];
      kappal = kappastore[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)];
      cpl = sqrt( kappal / rhol );
            
      // gets associated, weighted jacobian      
      jacobianw = abs_boundary_jacobian2Dw[INDEX2(NGLL2,igll,iface)];            

//daniel
//if( igll == 0 ) printf("gpu: %i %i %i %i %i %e %e %e\n",i,j,k,ispec,iglob,rhol,kappal,jacobianw);
         
      // Sommerfeld condition
      atomicAdd(&potential_dot_dot_acoustic[iglob],-potential_dot_acoustic[iglob]*jacobianw/cpl/rhol);      
      
      // adjoint simulations
      if( SIMULATION_TYPE == 3 ){
        // Sommerfeld condition
        atomicAdd(&b_potential_dot_dot_acoustic[iglob],-b_absorb_potential[INDEX2(NGLL2,igll,iface)]);
      }else if( SIMULATION_TYPE == 1 && SAVE_FORWARD == 1 ){  
         b_absorb_potential[INDEX2(NGLL2,igll,iface)] = potential_dot_acoustic[iglob]*jacobianw/cpl/rhol; 
      }
      
    }
//  }

}

/* ----------------------------------------------------------------------------------------------- */

extern "C" 
void FC_FUNC_(compute_stacey_acoustic_cuda,
              COMPUTE_STACEY_ACOUSTIC_CUDA)(
                                    long* Mesh_pointer_f, 
                                    int* phase_is_innerf, 
                                    int* num_abs_boundary_facesf, 
                                    int* SIMULATION_TYPEf, 
                                    int* SAVE_FORWARDf,
                                    float* h_b_absorb_potential) {
TRACE("compute_stacey_acoustic_cuda");
  //double start_time = get_time();

  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  int phase_is_inner          = *phase_is_innerf;
  int num_abs_boundary_faces  = *num_abs_boundary_facesf;
  int SIMULATION_TYPE         = *SIMULATION_TYPEf;
  int SAVE_FORWARD            = *SAVE_FORWARDf;              

  // way 1: Elapsed time: 4.385948e-03
  // > NGLLSQUARE==NGLL2==25, but we handle this inside kernel
  //  int blocksize = 32; 
  
  // way 2: Elapsed time: 4.379034e-03
  // > NGLLSQUARE==NGLL2==25, no further check inside kernel
  int blocksize = 25; 
  
  int num_blocks_x = num_abs_boundary_faces;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(blocksize,1,1);

  //  adjoint simulations: reads in absorbing boundary
  if (SIMULATION_TYPE == 3 && num_abs_boundary_faces > 0 ){  
    // copies array to GPU
    print_CUDA_error_if_any(cudaMemcpy(mp->d_b_absorb_potential,h_b_absorb_potential,
                                       mp->d_b_reclen_potential,cudaMemcpyHostToDevice),700);    
  }
      
  compute_stacey_acoustic_kernel<<<grid,threads>>>(mp->d_potential_dot_acoustic,
                                                   mp->d_potential_dot_dot_acoustic,
                                                   mp->d_abs_boundary_ispec, 
                                                   mp->d_abs_boundary_ijk, 
                                                   mp->d_ibool, 
                                                   mp->d_rhostore, 
                                                   mp->d_kappastore, 
                                                   mp->d_abs_boundary_jacobian2Dw, 
                                                   mp->d_ispec_is_inner, 
                                                   mp->d_ispec_is_acoustic, 
                                                   phase_is_inner,
                                                   SIMULATION_TYPE,SAVE_FORWARD,
                                                   mp->d_b_potential_dot_acoustic,
                                                   mp->d_b_potential_dot_dot_acoustic,
                                                   mp->d_b_absorb_potential);
  
  //  adjoint simulations: stores absorbed wavefield part
  if (SIMULATION_TYPE == 1 && SAVE_FORWARD == 1 && num_abs_boundary_faces > 0 ){  
    // copies array to CPU  
    print_CUDA_error_if_any(cudaMemcpy(h_b_absorb_potential,mp->d_b_absorb_potential,
                                       mp->d_b_reclen_potential,cudaMemcpyDeviceToHost),701);
  }
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING  
  exit_on_cuda_error("compute_stacey_acoustic_kernel");
#endif
}

