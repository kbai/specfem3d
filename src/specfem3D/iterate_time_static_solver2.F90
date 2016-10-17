#define XPOSITION 250000.0e0_CUSTOM_REAL
#define YPOSITION 250000.0e0_CUSTOM_REAL
#define ZPOSITION 0.0e0_CUSTOM_REAL
#define IT 1024
subroutine iterate_time_static_solver2()

    use conjugate_gradient
    use specfem_par
    use specfem_par_acoustic
    use specfem_par_elastic
    use specfem_par_poroelastic
    use specfem_par_movie
    use gravity_perturbation, only : gravity_timeseries, GRAVITY_SIMULATION
    use fault_solver_dynamic, only : bc_dynflt_set3d_all,SIMULATION_TYPE_DYN,faults, synchronize_gpu
    use fault_solver_kinematic, only : bc_kinflt_set_all,SIMULATION_TYPE_KIN
    !use fault_solver_qstatic, only: bc_qstaticflt_set3d_all,faults


    implicit none

    type(CG_data) ::  CGC,CG
    type(CG_Vector) :: CG_size
    logical,dimension(3,NGLOB_AB) :: MASK_default
    real(kind=CUSTOM_REAL) :: Max_error
    real(kind=CUSTOM_REAL) :: Max_error_all
    real(kind=CUSTOM_REAL),dimension(3,NGLOB_AB) :: sload,displ2,displ3,precon
    integer :: max_loc,ii
    integer :: NGLOB_AB_ALL
    integer,dimension(1) :: Boundary_loc

    double precision :: start, finish
    real(kind=CUSTOM_REAL) :: dtsngl

!    write(*,*) "ready to compute diagonal!"
!    write(IMAIN,*) "xigll",xigll
!    write(IMAIN,*) "wxgll",wxgll
    dtsngl = sngl(dt)
    open(unit=IT,file=trim(OUTPUT_FILES)//'/output_iterative.txt',status='unknown') 
    call print_computer_name()
    call make_load()
    call get_boundary_free_surface()
    call prepare_MPI()
    call prepare_restri()
    call make_displacement()
!    call write_movie_output()
!    write(*,*) "MPI done!"
    call prepare_CGC()
! initialize the conjugate gradient solver for the sparse grid
    call prepare_CG()
! initialize the conjugate gradient solver on regular grid.
    start = wtime()   
    do it = 1,3000
    !call update_value_direction(CG)
    dtsngl = 0.001; 
    write(*,*) 'deltat_fortran:',dt
    !call update_fault_displ_static(Mesh_pointer, Fault_pointer, dt, myrank, it)
    call it_update_displacement_static_cuda(Mesh_pointer,dtsngl)

   
    call compute_force_on_fault(Mesh_pointer, Fault_pointer, myrank, it)
!    call fault_solver_gpu(Mesh_pointer,Fault_pointer,dtsngl,myrank,it)
!    if(mod(it,500)==0) call synchronize_gpu(it)

    call update_solution(Mesh_pointer, CG_pointer, myrank,dtsngl)
    call it_update_displacement2_static_cuda(Mesh_pointer,dtsngl)

    enddo 
    call write_movie_output()


    finish = wtime()

    write(*,*) 'cpu-time',finish-start

contains
subroutine make_load()
     time_start = wtime()
     load(:,:)=0.0_CUSTOM_REAL
     MASK_default(:,:) = .true.

       write(*,*) 'fault1 nodes',faults(1)%nglob
        if(faults(1)%nglob>0) then
            load(:,faults(1)%ibulk1) = 0.0_CUSTOM_REAL
            load(:,faults(1)%ibulk2) = 0.0_CUSTOM_REAL
            load(1,faults(1)%ibulk1) = load(1,faults(1)%ibulk1)+10.0e6_CUSTOM_REAL*faults(1)%B(:)
            load(1,faults(1)%ibulk2) = load(1,faults(1)%ibulk2)-10.0e6_CUSTOM_REAL*faults(1)%B(:)
!                    write(*,*) "sum of B!",myrank,sum(faults(1)%B(:))

        endif
!        write(IMAIN,*) maxval(load)
         call restriction_call(sload,load,MASK_default,MASK_default)
         CG_size%NDIM = 3
         CG_size%NELE = NGLOB_AB
end subroutine make_load
!================================================================================
subroutine prepare_CGC()
    use specfem_par
 
    displ2 = 0.0_CUSTOM_REAL
    if(myrank == 11) write(*,*) "true of false?",MASK_default(1,26177)
    write(*,*) "sizeofarray", size(MASK_default)
    write(*,*) "sizeofarray", size(load)
    !call CG_initialize(CGC,CG_size,displ,sload,.true.,MASK_default,MASK_default)
!    displ(:,1) = 3.1415926;
    call compute_Diagonal(precon);
    write(*,*) "maxload=",maxval(load)
    call passhandle(Mesh_pointer, CG_pointer, NGLOB_AB, displ, load, sload, MASK_default,MASK_default,myrank,MPI_repeat,restri_op)
    call write_movie_output()
   
    if(myrank == 31) write(*,*),'gpu force:',load(1:3,1:10)

end subroutine prepare_CGC

!================================================================================


subroutine prepare_CG()
   displ(:,:) = 0.0_CUSTOM_REAL
   call CG_initialize(CG,CG_size,displ,load,.FALSE.,MASK_default,MASK_default)
   call compute_Diagonal(precon)
   rmassx = 1.0_CUSTOM_REAL/precon(1,:)
   rmassy = 1.0_CUSTOM_REAL/precon(2,:)
   rmassz = 1.0_CUSTOM_REAL/precon(3,:)

   call CG_initialize_preconditioner(CG,rmassx,rmassy,rmassz)
end subroutine prepare_CG

!=================================================================================

subroutine get_boundary()
integer igll,counter,ispec,i,j,k
    do counter = 1,num_abs_boundary_faces
        ispec = abs_boundary_ispec(counter)
        do igll = 1,NGLLSQUARE
            i = abs_boundary_ijk(1,igll,counter)
            j = abs_boundary_ijk(2,igll,counter)
            k = abs_boundary_ijk(3,igll,counter)
            MASK_default(:,ibool(i,j,k,ispec)) = .false.
        enddo
    enddo
end subroutine get_boundary

subroutine get_boundary_free_surface()
integer igll,counter,ispec,i,j,k
   if(myrank == 0) then
    do counter = 1,1
    write(*,*) 'set 1 fixed element!',num_abs_boundary_faces
        ispec = abs_boundary_ispec(counter)
    do igll = 1,1
        i = abs_boundary_ijk(1,igll,counter)
        j = abs_boundary_ijk(2,igll,counter)
        k = abs_boundary_ijk(3,igll,counter)
        MASK_default(:,ibool(i,j,k,ispec)) = .false.
    enddo
    enddo
endif

end subroutine get_boundary_free_surface


end subroutine iterate_time_static_solver2

subroutine make_displacement()
    use specfem_par
    use specfem_par_elastic

!    displ(:,:) = 10.0

    displ(1,:) = exp(-1e-6*((xstore - 10.0e3)*(xstore- 10.0e3) + (ystore - 10.0e3)*(ystore - 10.0e3) + (zstore + 10.0e3)*(zstore + 10.0e3))) 



end subroutine make_displacement


!subroutine vcycle()
!        call CG_initialize(CG,CG_size,displ2,load,.FALSE.,MASK_default,MASK_default)
!        call update_value_direction(CG)
 
