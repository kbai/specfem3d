#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <mpi.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "mesh_constants_cuda.h"

typedef float real; //type of variables passed into function
typedef float realw; //type of "working" variables

#define MAXDEBUG 1
#define ENABLE_VERY_SLOW_ERROR_CHECKING

#if MAXDEBUG == 1
#define LOG(x) printf("%s\n",x)
#define PRINT5(var,offset) for(;print_count<5;print_count++) printf("var(%d)=%2.20f\n",print_count,var[offset+print_count]);
#define PRINT10(var) if(print_count<10) { printf("var=%1.20e\n",var); print_count++; }
#define PRINT10i(var) if(print_count<10) { printf("var=%d\n",var); print_count++; }
#else
#define LOG(x) // printf("%s\n",x);
#define PRINT5(var,offset) // for(i=0;i<10;i++) printf("var(%d)=%f\n",i,var[offset+i]);
#endif

#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + (y)*xsize + (z)*xsize*ysize
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize
#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + (y)*xsize + (z)*xsize*ysize + (i)*xsize*ysize*zsize + (j)*xsize*ysize*zsize*isize

#define NDIM 3
#define NGLLX 5

__global__ void compute_stacey_elastic_kernel(real* veloc, real* accel, real* b_accel, int* abs_boundary_ispec,
					      int* abs_boundary_ijk, int* ibool,
					      real* abs_boundary_normal,
					      real* rho_vp, real* rho_vs,
					      real* abs_boundary_jacobian2Dw,
					      real* b_absorb_field,int NGLLSQUARE,
					      int* ispec_is_inner, int* ispec_is_elastic,
					      int phase_is_inner,float* debug_val,int* debug_val_int,
					      int num_abs_boundary_faces,
					      int SAVE_FORWARD,int SIMULATION_TYPE) {

  int igll = threadIdx.x; // tx
  int iface = blockIdx.x + gridDim.x*blockIdx.y; // bx
  int i;
  int j;
  int k;
  int iglob;
  int ispec;
  realw vx,vy,vz,vn;
  realw nx,ny,nz;
  realw rho_vp_temp,rho_vs_temp;
  realw tx,ty,tz;
  realw jacobianw;
  // don't compute points outside NGLLSQUARE=25


  
  if(igll<NGLLSQUARE && iface < num_abs_boundary_faces) {    
    
    // "-1" from index values to convert from Fortran-> C indexing
    ispec = abs_boundary_ispec[iface]-1;
    i = abs_boundary_ijk[INDEX3(NDIM,NGLLSQUARE,0,igll,iface)]-1;
    j = abs_boundary_ijk[INDEX3(NDIM,NGLLSQUARE,1,igll,iface)]-1;
    k = abs_boundary_ijk[INDEX3(NDIM,NGLLSQUARE,2,igll,iface)]-1;
    iglob = ibool[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)]-1;
    
    if(ispec_is_inner[ispec] == phase_is_inner && ispec_is_elastic[ispec]==1) {

      i = abs_boundary_ijk[INDEX3(NDIM,NGLLSQUARE,0,igll,iface)]-1;
      j = abs_boundary_ijk[INDEX3(NDIM,NGLLSQUARE,1,igll,iface)]-1;
      k = abs_boundary_ijk[INDEX3(NDIM,NGLLSQUARE,2,igll,iface)]-1;
      iglob = ibool[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)]-1;
      
      // gets associated velocity
      
      vx = veloc[iglob*3+0];
      vy = veloc[iglob*3+1];
      vz = veloc[iglob*3+2];
      
      // gets associated normal
      nx = abs_boundary_normal[INDEX3(NDIM,NGLLSQUARE,0,igll,iface)];
      ny = abs_boundary_normal[INDEX3(NDIM,NGLLSQUARE,1,igll,iface)];
      nz = abs_boundary_normal[INDEX3(NDIM,NGLLSQUARE,2,igll,iface)];
      
      // // velocity component in normal direction (normal points out of element)
      vn = vx*nx + vy*ny + vz*nz;
      rho_vp_temp = rho_vp[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)];
      rho_vs_temp = rho_vs[INDEX4(NGLLX,NGLLX,NGLLX,i,j,k,ispec)];
      tx = rho_vp_temp*vn*nx + rho_vs_temp*(vx-vn*nx);
      ty = rho_vp_temp*vn*ny + rho_vs_temp*(vy-vn*ny);
      tz = rho_vp_temp*vn*nz + rho_vs_temp*(vz-vn*nz);
      
      jacobianw = abs_boundary_jacobian2Dw[INDEX2(NGLLSQUARE,igll,iface)];            
   
      atomicAdd(&accel[iglob*3],-tx*jacobianw);
      atomicAdd(&accel[iglob*3+1],-ty*jacobianw);
      atomicAdd(&accel[iglob*3+2],-tz*jacobianw);

      if(SIMULATION_TYPE == 3) {
	atomicAdd(&b_accel[iglob*3  ],-b_absorb_field[0+3*(igll+25*(iface))]);
	atomicAdd(&b_accel[iglob*3+1],-b_absorb_field[1+3*(igll+25*(iface))]);
	atomicAdd(&b_accel[iglob*3+2],-b_absorb_field[2+3*(igll+25*(iface))]);
      }
      else if(SAVE_FORWARD && SIMULATION_TYPE == 1) {
	b_absorb_field[0+3*(igll+25*(iface))] = tx*jacobianw;
	b_absorb_field[1+3*(igll+25*(iface))] = ty*jacobianw;
	b_absorb_field[2+3*(igll+25*(iface))] = tz*jacobianw;
      }
      
    }
  }

}

#define FC_FUNC(name,NAME) name ## _
#define FC_FUNC_(name,NAME) name ## _

extern "C" void
FC_FUNC_(write_abs,WRITE_ABS)(int *fid, char *buffer, int *length , int *index);
extern "C" void
FC_FUNC_(read_abs,READ_ABS)(int *fid, char *buffer, int *length , int *index);

extern "C" void compute_stacey_elastic_cuda_(long* Mesh_pointer_f, int* NSPEC_ABf, int* NGLOB_ABf, int* phase_is_innerf, int* num_abs_boundary_facesf, int* SIMULATION_TYPEf, int* NSTEPf, int* NGLOB_ADJOINTf, int* b_num_abs_boundary_facesf, int* b_reclen_fieldf,float* b_absorb_field, int* SAVE_FORWARDf, int* NGLLSQUAREf,int* itf) {
  
  Mesh* mp = (Mesh*)(*Mesh_pointer_f); //get mesh pointer out of fortran integer container
  int fid = 0;
  int it = *itf;
  int NSPEC_AB = *NSPEC_ABf;
  int NGLOB_AB = *NGLOB_ABf;
  int NGLLSQUARE = *NGLLSQUAREf;
  int phase_is_inner	     = *phase_is_innerf;
  int num_abs_boundary_faces     = *num_abs_boundary_facesf;
  int SIMULATION_TYPE	     = *SIMULATION_TYPEf;
  int NSTEP			     = *NSTEPf;
  int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  int NGLOB_ADJOINT		     = *NGLOB_ADJOINTf;
  int b_num_abs_boundary_faces   = *b_num_abs_boundary_facesf;
  int b_reclen_field	     = *b_reclen_fieldf;
  int SAVE_FORWARD             = *SAVE_FORWARDf;              

  int blocksize = 32; // > NGLLSQUARE=25, but we handle this inside kernel
  int num_blocks_x = num_abs_boundary_faces;
  int num_blocks_y = 1;
  while(num_blocks_x > 65535) {
    num_blocks_x = ceil(num_blocks_x/2.0);
    num_blocks_y = num_blocks_y*2;
  }

  dim3 grid(num_blocks_x,num_blocks_y);
  dim3 threads(blocksize,1,1);

  float* d_debug_val;
  int* d_debug_val_int;

  if(SIMULATION_TYPE == 3 && num_abs_boundary_faces > 0) {
    // int val = NSTEP-it+1;
    // read_abs_(&fid,(char*)b_absorb_field,&b_reclen_field,&val);    
    // The read is done in fortran
    cudaMemcpy(mp->d_b_absorb_field,b_absorb_field,b_reclen_field,cudaMemcpyHostToDevice);
  }
  
  #ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("between cudamemcpy and compute_stacey_elastic_kernel");
  #endif
  
  compute_stacey_elastic_kernel<<<grid,threads>>>(mp->d_veloc,mp->d_accel,mp->d_b_accel,mp->d_abs_boundary_ispec, mp->d_abs_boundary_ijk, mp->d_ibool, mp->d_abs_boundary_normal, mp->d_rho_vp, mp->d_rho_vs, mp->d_abs_boundary_jacobian2Dw, mp->d_b_absorb_field,NGLLSQUARE,mp->d_ispec_is_inner, mp->d_ispec_is_elastic, phase_is_inner,d_debug_val,d_debug_val_int,num_abs_boundary_faces,SAVE_FORWARD,SIMULATION_TYPE);
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("compute_stacey_elastic_kernel");  
#endif

  // ! adjoint simulations: stores absorbed wavefield part
  // if (SIMULATION_TYPE == 1 .and. SAVE_FORWARD .and. num_abs_boundary_faces > 0 ) &
  //   write(IOABS,rec=it) b_reclen_field,b_absorb_field,b_reclen_field
  
  if(SIMULATION_TYPE==1 && SAVE_FORWARD && num_abs_boundary_faces>0) {
    cudaMemcpy(b_absorb_field,mp->d_b_absorb_field,b_reclen_field,cudaMemcpyDeviceToHost);
    // The write is done in fortran
    // write_abs_(&fid,(char*)b_absorb_field,&b_reclen_field,&it);    
  }
  
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_cuda_error("after compute_stacey_elastic after cudamemcpy");  
#endif
}

