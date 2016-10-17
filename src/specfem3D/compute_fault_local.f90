subroutine compute_fault_GPU_local(AX,X,MASKX,MASKAX)

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
  enddo


end subroutine compute_fault_GPU_local


