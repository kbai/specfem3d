subroutine compute_fault_GPU(AX,X,MASKX,MASKAX)

  use specfem_par
  use specfem_par_elastic

  implicit none

  integer:: iphase
  logical:: phase_is_inner
  real(kind=CUSTOM_REAL),dimension(3,NGLOB_AB),intent(in) :: X
  real(kind=CUSTOM_REAL),dimension(3,NGLOB_AB),intent(out) :: AX
!  logical,dimension(3,NGLOB_AB),optional :: MASKXin,MASKAXin
  logical,dimension(3,NGLOB_AB),intent(in) :: MASKX,MASKAX
 
  ! distinguishes two runs: for points on MPI interfaces, and points within the partitions
  do iphase=1,2

    !first for points on MPI interfaces
    if( iphase == 1 ) then
      phase_is_inner = .false.
    else
      phase_is_inner = .true.
    endif

    ! elastic term
    ! contains both forward SIM_TYPE==1 and backward SIM_TYPE==3 simulations
    call compute_forces_fault(Mesh_pointer, iphase, deltat, &
                                          X,AX,MASKX,MASKAX, &
                                          nspec_outer_elastic, &
                                          nspec_inner_elastic, &
                                          myrank)

    ! while inner elements compute "Kernel_2", we wait for MPI to
    ! finish and transfer the boundary terms to the device asynchronously
    if(phase_is_inner .eqv. .true.) then
      !daniel: todo - this avoids calling the fortran vector send from CUDA routine
      ! wait for asynchronous copy to finish
      call sync_copy_from_device(Mesh_pointer,iphase,buffer_send_vector_ext_mesh)

      ! sends mpi buffers
      call assemble_MPI_vector_send_cuda(NPROC, &
                  buffer_send_vector_ext_mesh,buffer_recv_vector_ext_mesh, &
                  num_interfaces_ext_mesh,max_nibool_interfaces_ext_mesh, &
                  nibool_interfaces_ext_mesh,&
                  my_neighbours_ext_mesh, &
                  request_send_vector_ext_mesh,request_recv_vector_ext_mesh)

      ! transfers mpi buffers onto GPU
      call transfer_boundary_to_device(NPROC,Mesh_pointer,buffer_recv_vector_ext_mesh, &
                  num_interfaces_ext_mesh,max_nibool_interfaces_ext_mesh, &
                  request_recv_vector_ext_mesh)
    endif ! inner elements

    ! assemble all the contributions between slices using MPI
    if( phase_is_inner .eqv. .false. ) then
      call transfer_boundary_from_device_a(Mesh_pointer,nspec_outer_elastic)

    else
      ! waits for send/receive requests to be completed and assembles values
      call assemble_MPI_vector_write_cuda(NPROC,NGLOB_AB,accel, Mesh_pointer,&
                      buffer_recv_vector_ext_mesh,num_interfaces_ext_mesh,&
                      max_nibool_interfaces_ext_mesh, &
                      nibool_interfaces_ext_mesh,ibool_interfaces_ext_mesh, &
                      request_send_vector_ext_mesh,request_recv_vector_ext_mesh, &
                      1)
    endif

  enddo


end subroutine compute_fault_GPU


