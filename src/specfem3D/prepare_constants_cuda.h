#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H
/* CUDA specific things from specfem3D_kernels.cu */

#define NGLL2 25


#ifdef USE_TEXTURES
// declaration of textures
texture<float, 1, cudaReadModeElementType> tex_displ;
texture<float, 1, cudaReadModeElementType> tex_accel;

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

#endif

// setters for these const arrays (very ugly hack, but will have to do)

void setConst_hprime_xx(float* array);

void setConst_hprimewgll_xx(float* array);
  
void setConst_wgllwgll_xy(float* array,Mesh* mp);

void setConst_wgllwgll_xz(float* array, Mesh* mp);

void setConst_wgllwgll_yz(float* array, Mesh* mp);
void exit_on_cuda_error(char* kernel_name);

void show_free_memory(char* info_str);

void print_CUDA_error_if_any(cudaError_t err, int num)
{
  if (cudaSuccess != err)
  {
    printf("\nCUDA error !!!!! <%s> !!!!! at CUDA call # %d\n",cudaGetErrorString(err),num);
    pause_for_debugger(1);
    show_free_memory("after error\n");
    fflush(stdout);
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    exit(0);
  }
  return;
}
  







#endif //CUDA_HEADER_H
