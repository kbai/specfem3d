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

#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

/* CUDA specific things from specfem3D_kernels.cu */

#ifdef USE_TEXTURES
  // declaration of textures
  texture<float, 1, cudaReadModeElementType> tex_displ;
  texture<float, 1, cudaReadModeElementType> tex_accel;

  texture<float, 1, cudaReadModeElementType> tex_potential_acoustic;
  texture<float, 1, cudaReadModeElementType> tex_potential_dot_dot_acoustic;

  // for binding the textures

  void bindTexturesDispl(float* d_displ)
  {
    cudaError_t err;

    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();

    err = cudaBindTexture(NULL,tex_displ, d_displ, channelDescFloat, NDIM*NGLOB*sizeof(float));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in bindTexturesDispl for displ: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }

  void bindTexturesAccel(float* d_accel)
  {
    cudaError_t err;

    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();

    err = cudaBindTexture(NULL,tex_accel, d_accel, channelDescFloat, NDIM*NGLOB*sizeof(float));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in bindTexturesAccel for accel: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }

  void bindTexturesPotential(float* d_potential_acoustic)
  {
    cudaError_t err;
    
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
    
    err = cudaBindTexture(NULL,tex_potential_acoustic, d_potential_acoustic, 
                          channelDescFloat, NGLOB*sizeof(float));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in bindTexturesPotential for potential_acoustic: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }

  void bindTexturesPotential_dot_dot(float* d_potential_dot_dot_acoustic)
  {
    cudaError_t err;
    
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
    
    err = cudaBindTexture(NULL,tex_potential_dot_dot_acoustic, d_potential_dot_dot_acoustic, 
                          channelDescFloat, NGLOB*sizeof(float));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in bindTexturesPotential_dot_dot for potential_dot_dot_acoustic: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }

#endif

/* ----------------------------------------------------------------------------------------------- */

// setters for these const arrays (very ugly hack, but will have to do)

// elastic
void setConst_hprime_xx(float* array,Mesh* mp);
void setConst_hprime_yy(float* array,Mesh* mp);
void setConst_hprime_zz(float* array,Mesh* mp);

void setConst_hprimewgll_xx(float* array,Mesh* mp);

void setConst_wgllwgll_xy(float* array,Mesh* mp);
void setConst_wgllwgll_xz(float* array, Mesh* mp);
void setConst_wgllwgll_yz(float* array, Mesh* mp);


#endif //CUDA_HEADER_H
