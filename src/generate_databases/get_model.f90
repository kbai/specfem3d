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


! wrapper function

  subroutine get_model(myrank,nspec,ibool,mat_ext_mesh,nelmnts_ext_mesh, &
                        materials_ext_mesh,nmat_ext_mesh, &
                        undef_mat_prop,nundefMat_ext_mesh, &
                        ANISOTROPY,LOCAL_PATH)

  use generate_databases_par,only: IMODEL
  use create_regions_mesh_ext_par
  implicit none

  ! number of spectral elements in each block
  integer :: myrank,nspec
  integer, dimension(NGLLX,NGLLY,NGLLZ,nspec) :: ibool
  ! external mesh
  integer :: nelmnts_ext_mesh
  integer :: nmat_ext_mesh,nundefMat_ext_mesh
  integer, dimension(2,nelmnts_ext_mesh) :: mat_ext_mesh
  double precision, dimension(6,nmat_ext_mesh) :: materials_ext_mesh
  character (len=30), dimension(6,nundefMat_ext_mesh):: undef_mat_prop
  ! anisotropy
  logical :: ANISOTROPY
  character(len=256) LOCAL_PATH

  !----------------------------------------------------------
  ! USER Parameter
  ! daniel: TODO -- uses Piero's get_model_PREM routine rather than default one
  logical,parameter :: USE_PIERO_MODEL = .true.

  !----------------------------------------------------------

  if( USE_PIERO_MODEL ) then
    ! Using PREM model instead
    call get_model_PREM(myrank,nspec,ibool,mat_ext_mesh,nelmnts_ext_mesh, &
                        materials_ext_mesh,nmat_ext_mesh, &
                        undef_mat_prop,nundefMat_ext_mesh, &
                        ANISOTROPY,LOCAL_PATH)

  else
    ! DEFAULT routine
    call get_model_d(myrank,nspec,ibool,mat_ext_mesh,nelmnts_ext_mesh, &
                    materials_ext_mesh,nmat_ext_mesh, &
                    undef_mat_prop,nundefMat_ext_mesh, &
                    ANISOTROPY,LOCAL_PATH)
  endif

  end subroutine get_model

  !
  !-----------------------------------------------------------------------------------------------
  !

! default get model routine

  subroutine get_model_d(myrank,nspec,ibool,mat_ext_mesh,nelmnts_ext_mesh, &
                        materials_ext_mesh,nmat_ext_mesh, &
                        undef_mat_prop,nundefMat_ext_mesh, &
                        ANISOTROPY,LOCAL_PATH)


  use create_regions_mesh_ext_par
  implicit none

  ! number of spectral elements in each block
  integer :: myrank,nspec

  integer, dimension(NGLLX,NGLLY,NGLLZ,nspec) :: ibool

  ! external mesh
  integer :: nelmnts_ext_mesh
  integer :: nmat_ext_mesh,nundefMat_ext_mesh

  integer, dimension(2,nelmnts_ext_mesh) :: mat_ext_mesh
  double precision, dimension(16,nmat_ext_mesh) :: materials_ext_mesh
  character (len=30), dimension(6,nundefMat_ext_mesh):: undef_mat_prop

  ! anisotropy
  logical :: ANISOTROPY

  ! local parameters
  real(kind=CUSTOM_REAL) :: vp,vs,rho,qmu_atten
  real(kind=CUSTOM_REAL) :: c11,c12,c13,c14,c15,c16,c22,c23,c24,c25, &
                        c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66
  real(kind=CUSTOM_REAL) :: kappa_s,kappa_f,kappa_fr,mu_fr,rho_s,rho_f,phi,tort,eta_f, &
                        kxx,kxy,kxz,kyy,kyz,kzz,rho_bar
  real(kind=CUSTOM_REAL) :: cpIsquare,cpIIsquare,cssquare,H_biot,M_biot,C_biot,D_biot, &
                        afactor,bfactor,cfactor

  integer :: ispec,i,j,k
  
  ! material domain
  integer :: idomain_id
  
  integer :: imaterial_id,imaterial_def

  ! gll point location
  double precision :: xmesh,ymesh,zmesh  
  integer :: iglob
  character(len=256) LOCAL_PATH

  ! variables for importing models from files in SPECFEM format, e.g.,  proc000000_vp.bin etc.
  ! can be used for importing updated model in iterative inversions
  logical,parameter :: USE_EXTERNAL_FILES = .false.

  ! use acoustic domains for simulation
  logical,parameter :: USE_PURE_ACOUSTIC_MOD = .false.

  ! initializes element domain flags
  ispec_is_acoustic(:) = .false.
  ispec_is_elastic(:) = .false.
  ispec_is_poroelastic(:) = .false.

  ! prepares tomography model if needed for elements with undefined material definitions
  if( nundefMat_ext_mesh > 0 .or. IMODEL == IMODEL_TOMO ) then
    call model_tomography_broadcast(myrank)    
  endif

  ! prepares external model values if needed
  select case( IMODEL )
  case( IMODEL_USER_EXTERNAL )
    call model_external_broadcast(myrank)
  case( IMODEL_SALTON_TROUGH )
    call model_salton_trough_broadcast(myrank)
  end select

! !  Piero, read bedrock file
! in case, see file model_interface_bedrock.f90:
!  call model_bedrock_broadcast(myrank)


  ! material properties on all GLL points: taken from material values defined for
  ! each spectral element in input mesh
  do ispec = 1, nspec

    ! loops over all gll points in element
    do k = 1, NGLLZ
      do j = 1, NGLLY
        do i = 1, NGLLX

          ! initializes material
          vp = 0._CUSTOM_REAL
          vs = 0._CUSTOM_REAL
          rho = 0._CUSTOM_REAL
          
          rho_s = 0._CUSTOM_REAL
          kappa_s = 0._CUSTOM_REAL
          rho_f = 0._CUSTOM_REAL
          kappa_f = 0._CUSTOM_REAL
          eta_f = 0._CUSTOM_REAL
          kappa_fr = 0._CUSTOM_REAL
          mu_fr = 0._CUSTOM_REAL
          phi = 0._CUSTOM_REAL
          tort = 0._CUSTOM_REAL
          kxx = 0._CUSTOM_REAL
          kxy = 0._CUSTOM_REAL
          kxz = 0._CUSTOM_REAL
          kyy = 0._CUSTOM_REAL
          kyz = 0._CUSTOM_REAL
          kzz = 0._CUSTOM_REAL
          
          qmu_atten = 0._CUSTOM_REAL

          c11 = 0._CUSTOM_REAL
          c12 = 0._CUSTOM_REAL
          c13 = 0._CUSTOM_REAL
          c14 = 0._CUSTOM_REAL
          c15 = 0._CUSTOM_REAL
          c16 = 0._CUSTOM_REAL
          c22 = 0._CUSTOM_REAL
          c23 = 0._CUSTOM_REAL
          c24 = 0._CUSTOM_REAL
          c25 = 0._CUSTOM_REAL
          c26 = 0._CUSTOM_REAL
          c33 = 0._CUSTOM_REAL
          c34 = 0._CUSTOM_REAL
          c35 = 0._CUSTOM_REAL
          c36 = 0._CUSTOM_REAL
          c44 = 0._CUSTOM_REAL
          c45 = 0._CUSTOM_REAL
          c46 = 0._CUSTOM_REAL
          c55 = 0._CUSTOM_REAL
          c56 = 0._CUSTOM_REAL
          c66 = 0._CUSTOM_REAL

          ! gets xyz coordinates of GLL point
          iglob = ibool(i,j,k,ispec)
          xmesh = xstore_dummy(iglob)
          ymesh= ystore_dummy(iglob)
          zmesh = zstore_dummy(iglob)

          ! material index 1: associated material number
          ! 1 = acoustic, 2 = elastic, 3 = poroelastic, -1 = undefined tomographic        
          imaterial_id = mat_ext_mesh(1,ispec)

          ! material index 2: associated material definition
          ! 1 = interface, 2 = tomography material
          imaterial_def = mat_ext_mesh(2,ispec)

          ! assigns material properties
          call get_model_values(materials_ext_mesh,nmat_ext_mesh, &
                               undef_mat_prop,nundefMat_ext_mesh, &
                               imaterial_id,imaterial_def, &
                               xmesh,ymesh,zmesh, &
                               rho,vp,vs,qmu_atten,idomain_id, &
                               rho_s,kappa_s,rho_f,kappa_f,eta_f,kappa_fr,mu_fr, &
                               phi,tort,kxx,kxy,kxz,kyy,kyz,kzz, &
                               c11,c12,c13,c14,c15,c16, &
                               c22,c23,c24,c25,c26,c33, &
                               c34,c35,c36,c44,c45,c46,c55,c56,c66, &
                               ANISOTROPY)
          

          ! stores velocity model

          if(idomain_id == IDOMAIN_ACOUSTIC .or. idomain_id == IDOMAIN_ELASTIC) then 
          
            ! elastic or acoustic material

            ! density
            rhostore(i,j,k,ispec) = rho

            ! kappa, mu
            kappastore(i,j,k,ispec) = rho*( vp*vp - FOUR_THIRDS*vs*vs )
            mustore(i,j,k,ispec) = rho*vs*vs

            ! attenuation
            qmu_attenuation_store(i,j,k,ispec) = qmu_atten

            ! Stacey, a completer par la suite
            rho_vp(i,j,k,ispec) = rho*vp
            rho_vs(i,j,k,ispec) = rho*vs
            !
            rho_vpI(i,j,k,ispec) = rho*vp
            rho_vpII(i,j,k,ispec) = 0.d0
            rho_vsI(i,j,k,ispec) = rho*vs
            rhoarraystore(1,i,j,k,ispec) = rho
            rhoarraystore(2,i,j,k,ispec) = rho
            phistore(i,j,k,ispec) = 0.d0
            tortstore(i,j,k,ispec) = 1.d0
            !end pll

          else                                         
            
            ! poroelastic material

            ! solid properties
            rhoarraystore(1,i,j,k,ispec) = rho_s
            kappaarraystore(1,i,j,k,ispec) = kappa_s
            ! fluid properties
            rhoarraystore(2,i,j,k,ispec) = rho_f
            kappaarraystore(2,i,j,k,ispec) = kappa_f
            etastore(i,j,k,ispec) = eta_f
            ! frame properties
            kappaarraystore(3,i,j,k,ispec) = kappa_fr
            mustore(i,j,k,ispec) = mu_fr
            phistore(i,j,k,ispec) = phi
            tortstore(i,j,k,ispec) = tort
            permstore(1,i,j,k,ispec) = kxx
            permstore(2,i,j,k,ispec) = kxy
            permstore(3,i,j,k,ispec) = kxz
            permstore(4,i,j,k,ispec) = kyy
            permstore(5,i,j,k,ispec) = kyz
            permstore(6,i,j,k,ispec) = kzz

            !Biot coefficients for the input phi
            D_biot = kappa_s*(1._CUSTOM_REAL + phi*(kappa_s/kappa_f - 1._CUSTOM_REAL))
            H_biot = (kappa_s - kappa_fr)*(kappa_s - kappa_fr)/(D_biot - kappa_fr) &
                      + kappa_fr + 4._CUSTOM_REAL*mu_fr/3._CUSTOM_REAL
            C_biot = kappa_s*(kappa_s - kappa_fr)/(D_biot - kappa_fr)
            M_biot = kappa_s*kappa_s/(D_biot - kappa_fr)
            ! Approximated velocities (no viscous dissipation)
            rho_bar =  (1._CUSTOM_REAL - phi)*rho_s + phi*rho_f
            afactor = rho_bar - phi/tort*rho_f
            bfactor = H_biot + phi*rho_bar/(tort*rho_f)*M_biot - TWO*phi/tort*C_biot
            cfactor = phi/(tort*rho_f)*(H_biot*M_biot - C_biot*C_biot)
            cpIsquare = (bfactor + sqrt(bfactor*bfactor - 4._CUSTOM_REAL*afactor*cfactor))/(2._CUSTOM_REAL*afactor)
            cpIIsquare = (bfactor - sqrt(bfactor*bfactor - 4._CUSTOM_REAL*afactor*cfactor))/(2._CUSTOM_REAL*afactor)
            cssquare = mu_fr/afactor

            ! AC based on cpI,cpII & cs
            rho_vpI(i,j,k,ispec) = (rho_bar - phi/tort*rho_f)*sqrt(cpIsquare)
            rho_vpII(i,j,k,ispec) = (rho_bar - phi/tort*rho_f)*sqrt(cpIIsquare)
            rho_vsI(i,j,k,ispec) = (rho_bar - phi/tort*rho_f)*sqrt(cssquare)

          endif !if(idomain_id == IDOMAIN_ACOUSTIC .or. idomain_id == IDOMAIN_ELASTIC)

          ! stores anisotropic parameters
          if( ANISOTROPY ) then
            c11store(i,j,k,ispec) = c11
            c12store(i,j,k,ispec) = c12
            c13store(i,j,k,ispec) = c13
            c14store(i,j,k,ispec) = c14
            c15store(i,j,k,ispec) = c15
            c16store(i,j,k,ispec) = c16
            c22store(i,j,k,ispec) = c22
            c23store(i,j,k,ispec) = c23
            c24store(i,j,k,ispec) = c24
            c25store(i,j,k,ispec) = c25
            c26store(i,j,k,ispec) = c26
            c33store(i,j,k,ispec) = c33
            c34store(i,j,k,ispec) = c34
            c35store(i,j,k,ispec) = c35
            c36store(i,j,k,ispec) = c36
            c44store(i,j,k,ispec) = c44
            c45store(i,j,k,ispec) = c45
            c46store(i,j,k,ispec) = c46
            c55store(i,j,k,ispec) = c55
            c56store(i,j,k,ispec) = c56
            c66store(i,j,k,ispec) = c66
          endif


          ! stores material domain
          select case( idomain_id )
          case( IDOMAIN_ACOUSTIC )          
            ispec_is_acoustic(ispec) = .true.
          case( IDOMAIN_ELASTIC )
            ispec_is_elastic(ispec) = .true.
          case( IDOMAIN_POROELASTIC )           
            ispec_is_poroelastic(ispec) = .true.
          case default
            stop 'error material domain index'
          end select

        enddo
      enddo
    enddo
  enddo

  ! checks material domains
  do ispec=1,nspec
    ! checks if domain is set
    if( (ispec_is_acoustic(ispec) .eqv. .false.) &
          .and. (ispec_is_elastic(ispec) .eqv. .false.) &
          .and. (ispec_is_poroelastic(ispec) .eqv. .false.) ) then
      print*,'error material domain not assigned to element:',ispec
      print*,'acoustic: ',ispec_is_acoustic(ispec)
      print*,'elastic: ',ispec_is_elastic(ispec)
      print*,'poroelastic: ',ispec_is_poroelastic(ispec)
      stop 'error material domain index element'
    endif
    ! checks if domain is unique
    if( ((ispec_is_acoustic(ispec) .eqv. .true.) .and. (ispec_is_elastic(ispec) .eqv. .true.)) .or. &
       ((ispec_is_acoustic(ispec) .eqv. .true.) .and. (ispec_is_poroelastic(ispec) .eqv. .true.)) .or. &
       ((ispec_is_poroelastic(ispec) .eqv. .true.) .and. (ispec_is_elastic(ispec) .eqv. .true.)) .or. &
       ((ispec_is_acoustic(ispec) .eqv. .true.) .and. (ispec_is_elastic(ispec) .eqv. .true.) .and. &
       (ispec_is_poroelastic(ispec) .eqv. .true.)) ) then
      print*,'error material domain assigned twice to element:',ispec
      print*,'acoustic: ',ispec_is_acoustic(ispec)
      print*,'elastic: ',ispec_is_elastic(ispec)
      print*,'poroelastic: ',ispec_is_poroelastic(ispec)
      stop 'error material domain index element'
    endif
  enddo

  ! GLL model
  ! variables for importing models from files in SPECFEM format, e.g.,  proc000000_vp.bin etc.
  ! can be used for importing updated model in iterative inversions
  if( IMODEL == IMODEL_GLL ) then
    ! note:
    ! import the model from files in SPECFEM format
    ! note that those those files should be saved in LOCAL_PATH    
    call model_gll(myrank,nspec,LOCAL_PATH)
  endif

  end subroutine get_model

!
!-------------------------------------------------------------------------------------------------
!

  subroutine get_model_values(materials_ext_mesh,nmat_ext_mesh, &
                             undef_mat_prop,nundefMat_ext_mesh, &
                             imaterial_id,imaterial_def, &
                             xmesh,ymesh,zmesh, &
                             rho,vp,vs,qmu_atten,idomain_id, &
                             rho_s,kappa_s,rho_f,kappa_f,eta_f,kappa_fr,mu_fr, &
                             phi,tort,kxx,kxy,kxz,kyy,kyz,kzz, &
                             c11,c12,c13,c14,c15,c16, &
                             c22,c23,c24,c25,c26,c33, &
                             c34,c35,c36,c44,c45,c46,c55,c56,c66, &
                             ANISOTROPY)

  use generate_databases_par,only: IMODEL
  use create_regions_mesh_ext_par
  implicit none

  integer, intent(in) :: nmat_ext_mesh
  double precision, dimension(16,nmat_ext_mesh),intent(in) :: materials_ext_mesh

  integer, intent(in) :: nundefMat_ext_mesh
  character (len=30), dimension(6,nundefMat_ext_mesh):: undef_mat_prop

  integer, intent(in) :: imaterial_id,imaterial_def  

  double precision, intent(in) :: xmesh,ymesh,zmesh

  real(kind=CUSTOM_REAL) :: vp,vs,rho,qmu_atten

  integer :: idomain_id

  real(kind=CUSTOM_REAL) :: kappa_s,kappa_f,kappa_fr,mu_fr,rho_s,rho_f,phi,tort,eta_f, &
                           kxx,kxy,kxz,kyy,kyz,kzz

  real(kind=CUSTOM_REAL) :: c11,c12,c13,c14,c15,c16,c22,c23,c24,c25, &
                        c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66

  logical :: ANISOTROPY

  ! local parameters
  integer :: iflag_aniso
  
  ! use acoustic domains for simulation
  logical,parameter :: USE_PURE_ACOUSTIC_MOD = .false.

  ! initializes with default values
  iflag_aniso = 0
  idomain_id = IDOMAIN_ELASTIC
  
  ! selects chosen velocity model
  select case( IMODEL )

  case( IMODEL_DEFAULT, IMODEL_GLL )
    ! material values determined by mesh properties
    call model_default(materials_ext_mesh,nmat_ext_mesh, &
                          undef_mat_prop,nundefMat_ext_mesh, &
                          imaterial_id,imaterial_def, &
                          xmesh,ymesh,zmesh, &
                          rho,vp,vs, &
                          iflag_aniso,qmu_atten,idomain_id, &
                          rho_s,kappa_s,rho_f,kappa_f,eta_f,kappa_fr,mu_fr, &
                          phi,tort,kxx,kxy,kxz,kyy,kyz,kzz)
        
  case( IMODEL_1D_PREM )
    ! 1D model profile from PREM
    call model_1D_prem_iso(xmesh,ymesh,zmesh,rho,vp,vs,qmu_atten)
                      
  case( IMODEL_1D_CASCADIA )
    ! 1D model profile for Cascadia region
    call model_1D_cascadia(xmesh,ymesh,zmesh,rho,vp,vs,qmu_atten)

  end subroutine get_model_d

!
!-------------------------------------------------------------------------------------------------
!
  subroutine get_model_PREM(myrank,nspec,ibool,mat_ext_mesh,nelmnts_ext_mesh, &
                        materials_ext_mesh,nmat_ext_mesh, &
                        undef_mat_prop,nundefMat_ext_mesh, &
                        ANISOTROPY,LOCAL_PATH)

  use create_regions_mesh_ext_par
  implicit none

  ! number of spectral elements in each block
  integer :: myrank,nspec

  integer, dimension(NGLLX,NGLLY,NGLLZ,nspec) :: ibool

  ! external mesh
  integer :: nelmnts_ext_mesh
  integer :: nmat_ext_mesh,nundefMat_ext_mesh

  integer, dimension(2,nelmnts_ext_mesh) :: mat_ext_mesh
  double precision, dimension(6,nmat_ext_mesh) :: materials_ext_mesh
  character (len=30), dimension(6,nundefMat_ext_mesh):: undef_mat_prop

  ! anisotropy
  logical :: ANISOTROPY

  ! local parameters
  real(kind=CUSTOM_REAL) :: vp,vs,rho,qmu_atten
  real(kind=CUSTOM_REAL) :: c11,c12,c13,c14,c15,c16,c22,c23,c24,c25, &
                        c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66
  integer :: ispec,i,j,k,iundef,ier
  integer :: iflag,flag_below,flag_above
  integer :: iflag_aniso,idomain_id,imaterial_id

!PB
  integer :: imaterial_PB

  ! gll point location
  double precision :: xloc,yloc,zloc
  integer :: iglob
  character(len=256) LOCAL_PATH,prname_lp
  real, dimension(:,:,:,:),allocatable :: vp_read,vs_read,rho_read

  ! variables for importing models from files in SPECFEM format, e.g.,  proc000000_vp.bin etc.
  ! can be used for importing updated model in iterative inversions
  logical,parameter :: USE_EXTERNAL_FILES = .false.

  ! use acoustic domains for simulation
  logical,parameter :: USE_PURE_ACOUSTIC_MOD = .false.



  ! initializes element domain flags
  ispec_is_acoustic(:) = .false.
  ispec_is_elastic(:) = .false.
  ispec_is_poroelastic(:) = .false.

!PB AT PRESENT I DON'T NEED THIS CALL CAUSE I'M USING PREM MODEL

  ! prepares tomography model if needed for elements with undefined material definitions
!  if( nundefMat_ext_mesh > 0 ) then
!    call model_tomography_broadcast(myrank)
!  endif

  ! prepares external model values if needed
  if( USE_MODEL_EXTERNAL_VALUES ) then
    call model_external_broadcast(myrank)
  endif

! !  Piero, read bedrock file
! in case, see file model_interface_bedrock.f90:
!  call model_bedrock_broadcast(myrank)


  ! material properties on all GLL points: taken from material values defined for
  ! each spectral element in input mesh
  do ispec = 1, nspec
    do k = 1, NGLLZ
      do j = 1, NGLLY
        do i = 1, NGLLX

           ! material index 1: associated material number
           imaterial_id = mat_ext_mesh(1,ispec)

           ! check if the material is known or unknown
           if( imaterial_id > 0) then
              ! gets velocity model as specified by (cubit) mesh files

              ! density
              ! materials_ext_mesh format:
              ! #index1 = rho #index2 = vp #index3 = vs #index4 = Q_flag #index5 = 0
              rho = materials_ext_mesh(1,imaterial_id)

              ! isotropic values: vp, vs
              vp = materials_ext_mesh(2,imaterial_id)
              vs = materials_ext_mesh(3,imaterial_id)

              ! attenuation
              qmu_atten = materials_ext_mesh(4,imaterial_id)

              ! anisotropy
              iflag_aniso = materials_ext_mesh(5,imaterial_id)

              ! material domain_id
              idomain_id = materials_ext_mesh(6,imaterial_id)

           else if (mat_ext_mesh(2,ispec) == 1) then

              stop 'material: interface not implemented yet'

              do iundef = 1,nundefMat_ext_mesh
                 if(trim(undef_mat_prop(2,iundef)) == 'interface') then
                    read(undef_mat_prop(3,iundef),'(1i3)') flag_below
                    read(undef_mat_prop(4,iundef),'(1i3)') flag_above
                 endif
              enddo

              ! see file model_interface_bedrock.f90: routine interface()
              !call interface(iflag,flag_below,flag_above,ispec,nspec,i,j,k,xstore,ystore,zstore,ibedrock)

              ! dummy: takes 1. defined material
              iflag = 1
              rho = materials_ext_mesh(1,iflag)
              vp = materials_ext_mesh(2,iflag)
              vs = materials_ext_mesh(3,iflag)
              qmu_atten = materials_ext_mesh(4,iflag)
              iflag_aniso = materials_ext_mesh(5,iflag)
              idomain_id = materials_ext_mesh(6,iflag)

           else if ( mat_ext_mesh(2,ispec) == 2 ) then

              imaterial_PB = abs(imaterial_id)
              ! material definition undefined, uses definition from tomography model
              ! GLL point location
              iglob = ibool(i,j,k,ispec)
              xloc = xstore_dummy(iglob)
              yloc = ystore_dummy(iglob)
              zloc = zstore_dummy(iglob)

             call PREM_routine(xloc,yloc,zloc, &
                                  rho,vp,vs,imaterial_PB)


!PB COMMENTED THE CALL TO
!model_tomography SINCE I WANT
!EACH GLL POINT HAVING
!PB VELOCITY VALUE DEFINED BY THE
!PREM_routine
              ! gets model values from tomography file
!              call model_tomography(xloc,yloc,zloc, &
!                                  rho,vp,vs)

              qmu_atten = ATTENUATION_COMP_MAXIMUM   ! attenuation: arbitrary value, see maximum in constants.h
              iflag_aniso = 0   ! no anisotropy

              ! sets acoustic/elastic domain as given in materials properties
              iundef = - imaterial_id    ! iundef must be positive
              read(undef_mat_prop(6,iundef),*) idomain_id
              ! or
              !idomain_id = IDOMAIN_ELASTIC    ! forces to be elastic domain

!PB

           else

              stop 'material: not implemented yet'

           end if

           ! adds/gets velocity model as specified in model_external_values.f90
           if( USE_MODEL_EXTERNAL_VALUES ) then
             call model_external_values(i,j,k,ispec,idomain_id,imaterial_id, &
                            nspec,ibool, &
                            iflag_aniso,qmu_atten, &
                            rho,vp,vs, &
                            c11,c12,c13,c14,c15,c16, &
                            c22,c23,c24,c25,c26,c33, &
                            c34,c35,c36,c44,c45,c46, &
                            c55,c56,c66,ANISOTROPY)
           endif

           ! adds anisotropic default model
           if( ANISOTROPY .and. .not. USE_MODEL_EXTERNAL_VALUES ) then
             call model_aniso(iflag_aniso,rho,vp,vs,c11,c12,c13,c14,c15,c16, &
                     c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45, &
                     c46,c55,c56,c66)

           endif

           ! stores velocity model

           ! density
           rhostore(i,j,k,ispec) = rho

           ! kappa, mu
           kappastore(i,j,k,ispec) = rho*( vp*vp - FOUR_THIRDS*vs*vs )
           mustore(i,j,k,ispec) = rho*vs*vs

           ! attenuation
           qmu_attenuation_store(i,j,k,ispec) = qmu_atten

           ! Stacey, a completer par la suite
           rho_vp(i,j,k,ispec) = rho*vp
           rho_vs(i,j,k,ispec) = rho*vs
           !end pll

           ! adds anisotropic perturbation to vp, vs
           if( ANISOTROPY ) then
             !call model_aniso(iflag_aniso,rho,vp,vs,c11,c12,c13,c14,c15,c16, &
             !        c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66)
             c11store(i,j,k,ispec) = c11
             c12store(i,j,k,ispec) = c12
             c13store(i,j,k,ispec) = c13
             c14store(i,j,k,ispec) = c14
             c15store(i,j,k,ispec) = c15
             c16store(i,j,k,ispec) = c16
             c22store(i,j,k,ispec) = c22
             c23store(i,j,k,ispec) = c23
             c24store(i,j,k,ispec) = c24
             c25store(i,j,k,ispec) = c25
             c26store(i,j,k,ispec) = c26
             c33store(i,j,k,ispec) = c33
             c34store(i,j,k,ispec) = c34
             c35store(i,j,k,ispec) = c35
             c36store(i,j,k,ispec) = c36
             c44store(i,j,k,ispec) = c44
             c45store(i,j,k,ispec) = c45
             c46store(i,j,k,ispec) = c46
             c55store(i,j,k,ispec) = c55
             c56store(i,j,k,ispec) = c56
             c66store(i,j,k,ispec) = c66
           endif

           ! for pure acoustic simulations (a way of avoiding re-mesh, re-partition etc.)
           ! can be used to compare elastic & acoustic reflections in exploration seismology
           ! do NOT use it unless you are confident
           if( USE_PURE_ACOUSTIC_MOD ) then
             idomain_id = IDOMAIN_ACOUSTIC
           endif

           ! material domain
           !print*,'velocity model:',ispec,idomain_id
           if( idomain_id == IDOMAIN_ACOUSTIC ) then
             ispec_is_acoustic(ispec) = .true.
           else if( idomain_id == IDOMAIN_ELASTIC ) then
             ispec_is_elastic(ispec) = .true.
           else if( idomain_id == IDOMAIN_POROELASTIC ) then
             stop 'poroelastic material domain not implemented yet'
             ispec_is_poroelastic(ispec) = .true.
           else
             stop 'error material domain index'
           endif

        enddo
      enddo
    enddo
    !print*,myrank,'ispec:',ispec,'rho:',rhostore(1,1,1,ispec),'vp:',vpstore(1,1,1,ispec),'vs:',vsstore(1,1,1,ispec)
  enddo

  ! checks material domains
  do ispec=1,nspec
    ! checks if domain is set
    if( (ispec_is_acoustic(ispec) .eqv. .false.) &
          .and. (ispec_is_elastic(ispec) .eqv. .false.) &
          .and. (ispec_is_poroelastic(ispec) .eqv. .false.) ) then
      print*,'error material domain not assigned to element:',ispec
      print*,'acoustic: ',ispec_is_acoustic(ispec)
      print*,'elastic: ',ispec_is_elastic(ispec)
      print*,'poroelastic: ',ispec_is_poroelastic(ispec)
      stop 'error material domain index element'
    endif
    ! checks if domain is unique
    if( ispec_is_acoustic(ispec) .eqv. .true. .and. ispec_is_elastic(ispec) .eqv. .true. ) then
      print*,'error material domain assigned twice to element:',ispec
      print*,'acoustic: ',ispec_is_acoustic(ispec)
      print*,'elastic: ',ispec_is_elastic(ispec)
      print*,'poroelastic: ',ispec_is_poroelastic(ispec)
      stop 'error material domain index element'
    endif
  enddo

! !! DK DK store the position of the six stations to be able to
! !! DK DK exclude circles around each station to make sure they are on the bedrock
! !! DK DK and not in the ice
! in case, see file model_interface_bedrock.f90: routine model_bedrock_store()


! import the model from files in SPECFEM format
! note that those those files should be saved in LOCAL_PATH

  if( USE_EXTERNAL_FILES ) then

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! if only vp structure is available (as is often the case in exploration seismology),
!!! use lines for vp only

    ! processors name
    write(prname_lp,'(a,i6.6,a)') trim(LOCAL_PATH)//'proc',myrank,'_'

    allocate( rho_read(NGLLX,NGLLY,NGLLZ,nspec),stat=ier)
    if( ier /= 0 ) stop 'error allocating array rho_read'
    write(prname_lp,'(a,i6.6,a)') trim(LOCAL_PATH)//'proc',myrank,'_'
    open(unit=28,file=prname_lp(1:len_trim(prname_lp))//'rho.bin',&
            status='unknown',action='read',form='unformatted')
    read(28) rho_read
    close(28)

    allocate( vp_read(NGLLX,NGLLY,NGLLZ,nspec),stat=ier)
    if( ier /= 0 ) stop 'error allocating array vp_read'
    write(prname_lp,'(a,i6.6,a)') trim(LOCAL_PATH)//'proc',myrank,'_'
    open(unit=28,file=prname_lp(1:len_trim(prname_lp))//'vp.bin',&
            status='unknown',action='read',form='unformatted')
    read(28) vp_read
    close(28)

    allocate( vs_read(NGLLX,NGLLY,NGLLZ,nspec),stat=ier)
    if( ier /= 0 ) stop 'error allocating array vs_read'
    write(prname_lp,'(a,i6.6,a)') trim(LOCAL_PATH)//'proc',myrank,'_'
    open(unit=28,file=prname_lp(1:len_trim(prname_lp))//'vs.bin',&
            status='unknown',action='read',form='unformatted')
    read(28) vs_read
    close(28)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! in cases where density structure is not given
!!! modify according to your desire

!  rho_read = 1000.0
!  where ( mustore > 100.0 )  &
!           rho_read = (1.6612 * (vp_read / 1000.0)     &
!                      -0.4720 * (vp_read / 1000.0)**2  &
!                      +0.0671 * (vp_read / 1000.0)**3  &
!                      -0.0043 * (vp_read / 1000.0)**4  &
!                      +0.000106*(vp_read / 1000.0)**5 )*1000.0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! in cases where shear wavespeed structure is not given
!!! modify according to your desire

!   vs_read = 0.0
!   where ( mustore > 100.0 )       vs_read = vp_read / sqrt(3.0)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! update arrays that will be saved and used in the solver xspecfem3D
!!! the following part is neccessary if you uncommented something above

    rhostore    = rho_read
    kappastore  = rhostore * ( vp_read * vp_read - FOUR_THIRDS * vs_read * vs_read )
    mustore     = rhostore * vs_read * vs_read
    rho_vp = rhostore * vp_read
    rho_vs = rhostore * vs_read

    ! free memory
    deallocate( rho_read,vp_read,vs_read)

  endif ! USE_EXTERNAL_FILES

  end subroutine get_model_PREM

