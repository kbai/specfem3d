!=====================================================================
!
!               S p e c f e m 3 D  V e r s i o n  1 . 4
!               ---------------------------------------
!
!                 Dimitri Komatitsch and Jeroen Tromp
!    Seismological Laboratory - California Institute of Technology
!         (c) California Institute of Technology September 2006
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

!   subroutine create_regions_mesh(xgrid,ygrid,zgrid,ibool,idoubling, &
!            xstore,ystore,zstore,npx,npy,iproc_xi,iproc_eta,addressing,nspec, &
!            volume_local,area_local_bottom,area_local_top, &
!            NGLOB_AB,npointot, &
!            NER_BOTTOM_MOHO,NER_MOHO_16,NER_16_BASEMENT,NER_BASEMENT_SEDIM,NER_SEDIM,NER, &
!            NEX_PER_PROC_XI,NEX_PER_PROC_ETA, &
!            NSPEC2DMAX_XMIN_XMAX,NSPEC2DMAX_YMIN_YMAX,NSPEC2D_BOTTOM,NSPEC2D_TOP, &
!            HARVARD_3D_GOCAD_MODEL,NPROC_XI,NPROC_ETA,NSPEC2D_A_XI,NSPEC2D_B_XI, &
!            NSPEC2D_A_ETA,NSPEC2D_B_ETA, &
!            myrank,LOCAL_PATH,UTM_X_MIN,UTM_X_MAX,UTM_Y_MIN,UTM_Y_MAX,Z_DEPTH_BLOCK,UTM_PROJECTION_ZONE, &
!            HAUKSSON_REGIONAL_MODEL,OCEANS, &
!            VP_MIN_GOCAD,VP_VS_RATIO_GOCAD_TOP,VP_VS_RATIO_GOCAD_BOTTOM, &
!            IMPOSE_MINIMUM_VP_GOCAD,THICKNESS_TAPER_BLOCK_HR,THICKNESS_TAPER_BLOCK_MR,MOHO_MAP_LUPEI, &
!            ANISOTROPY,SAVE_MESH_FILES,SUPPRESS_UTM_PROJECTION, &
!            ORIG_LAT_TOPO,ORIG_LONG_TOPO,DEGREES_PER_CELL_TOPO,NX_TOPO,NY_TOPO,USE_REGULAR_MESH)

  subroutine create_regions_mesh(xgrid,ygrid,zgrid,ibool,idoubling, &
           xstore,ystore,zstore,npx,npy,iproc_xi,iproc_eta,addressing,nspec, &
           NGLOB_AB,npointot, &
           NER_BOTTOM_MOHO,NER_MOHO_16,NER_16_BASEMENT,NER_BASEMENT_SEDIM,NER_SEDIM,NER, &
           NEX_PER_PROC_XI,NEX_PER_PROC_ETA, &
           NSPEC2DMAX_XMIN_XMAX,NSPEC2DMAX_YMIN_YMAX,NSPEC2D_BOTTOM,NSPEC2D_TOP, &
           HARVARD_3D_GOCAD_MODEL,NPROC_XI,NPROC_ETA,NSPEC2D_A_XI,NSPEC2D_B_XI, &
           NSPEC2D_A_ETA,NSPEC2D_B_ETA, &
           myrank,LOCAL_PATH,UTM_X_MIN,UTM_X_MAX,UTM_Y_MIN,UTM_Y_MAX,Z_DEPTH_BLOCK, &
           HAUKSSON_REGIONAL_MODEL,USE_REGULAR_MESH)

! create the different regions of the mesh

  implicit none

  include "constants.h"

! number of spectral elements in each block
  integer nspec

  integer NEX_PER_PROC_XI,NEX_PER_PROC_ETA!,UTM_PROJECTION_ZONE
  integer NER_BOTTOM_MOHO,NER_MOHO_16,NER_16_BASEMENT,NER_BASEMENT_SEDIM,NER_SEDIM,NER

  integer NSPEC2DMAX_XMIN_XMAX,NSPEC2DMAX_YMIN_YMAX,NSPEC2D_BOTTOM,NSPEC2D_TOP

  integer NPROC_XI,NPROC_ETA,NSPEC2D_A_XI,NSPEC2D_B_XI
  integer NSPEC2D_A_ETA,NSPEC2D_B_ETA
!  integer NX_TOPO,NY_TOPO

  integer npx,npy
  integer npointot

  logical HARVARD_3D_GOCAD_MODEL,HAUKSSON_REGIONAL_MODEL
  logical USE_REGULAR_MESH!,OCEANS,IMPOSE_MINIMUM_VP_GOCAD
!  logical MOHO_MAP_LUPEI,SUPPRESS_UTM_PROJECTION

  double precision UTM_X_MIN,UTM_X_MAX,UTM_Y_MIN,UTM_Y_MAX,Z_DEPTH_BLOCK
!  double precision VP_MIN_GOCAD,VP_VS_RATIO_GOCAD_TOP,VP_VS_RATIO_GOCAD_BOTTOM
  double precision horiz_size,vert_size!,THICKNESS_TAPER_BLOCK_HR,THICKNESS_TAPER_BLOCK_MR
!  double precision ORIG_LAT_TOPO,ORIG_LONG_TOPO,DEGREES_PER_CELL_TOPO

  character(len=150) LOCAL_PATH

  integer addressing(0:NPROC_XI-1,0:NPROC_ETA-1)

! arrays with the mesh
  double precision, dimension(NGLLX,NGLLY,NGLLZ,nspec) :: xstore,ystore,zstore
  real(kind=CUSTOM_REAL), dimension(:,:), allocatable :: nodes_coords

!   double precision xstore_local(NGLLX,NGLLY,NGLLZ)
!   double precision ystore_local(NGLLX,NGLLY,NGLLZ)
!   double precision zstore_local(NGLLX,NGLLY,NGLLZ)

  double precision xgrid(0:2*NER,0:2*NEX_PER_PROC_XI,0:2*NEX_PER_PROC_ETA)
  double precision ygrid(0:2*NER,0:2*NEX_PER_PROC_XI,0:2*NEX_PER_PROC_ETA)
  double precision zgrid(0:2*NER,0:2*NEX_PER_PROC_XI,0:2*NEX_PER_PROC_ETA)

  integer ibool(NGLLX,NGLLY,NGLLZ,nspec)

! use integer array to store topography values
!  integer icornerlat,icornerlong
!  double precision lat,long,elevation
!  double precision long_corner,lat_corner,ratio_xi,ratio_eta
!  integer itopo_bathy(NX_TOPO,NY_TOPO)

! auxiliary variables to generate the mesh
  integer ix,iy,iz,ir,ir1,ir2,dir
  integer ix1,ix2,dix,iy1,iy2,diy
  integer iax,iay,iar
  integer isubregion,nsubregions,doubling_index

! Gauss-Lobatto-Legendre points and weights of integration
!  double precision, dimension(:), allocatable :: xigll,yigll,zigll,wxgll,wygll,wzgll

! 3D shape functions and their derivatives
!   double precision, dimension(:,:,:,:), allocatable :: shape3D
!   double precision, dimension(:,:,:,:,:), allocatable :: dershape3D

! 2D shape functions and their derivatives
!   double precision, dimension(:,:,:), allocatable :: shape2D_x,shape2D_y,shape2D_bottom,shape2D_top
!   double precision, dimension(:,:,:,:), allocatable :: dershape2D_x,dershape2D_y,dershape2D_bottom,dershape2D_top

! topology of the elements
  integer iaddx(NGNOD)
  integer iaddy(NGNOD)
  integer iaddz(NGNOD)

  double precision xelm(NGNOD)
  double precision yelm(NGNOD)
  double precision zelm(NGNOD)

! parameters needed to store the radii of the grid points
! in the spherically symmetric Earth
  integer idoubling(nspec)

! ! for model density
!   real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: rhostore,kappastore,mustore
!   real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: c11store,c12store,c13store,c14store,c15store,c16store,&
!     c22store,c23store,c24store,c25store,c26store,c33store,c34store,c35store,c36store,c44store,c45store,c46store,&
!     c55store,c56store,c66store

! ! the jacobian
!   real(kind=CUSTOM_REAL) jacobianl

! boundary locator
  logical, dimension(:,:), allocatable :: iboun

! arrays with mesh parameters
!   real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: xixstore,xiystore,xizstore, &
!     etaxstore,etaystore,etazstore,gammaxstore,gammaystore,gammazstore,jacobianstore

! mass matrix and bathymetry for ocean load
!  integer ix_oceans,iy_oceans,iz_oceans,ispec_oceans
!  integer ispec2D_ocean_bottom
!  integer nglob_oceans
!  double precision xval,yval
!  double precision height_oceans
!  real(kind=CUSTOM_REAL), dimension(:), allocatable :: rmass_ocean_load

! proc numbers for MPI
  integer myrank

! check area and volume of the final mesh
!  double precision weight
!  double precision area_local_bottom,area_local_top
!  double precision volume_local

! variables for creating array ibool (some arrays also used for AVS or DX files)
  integer, dimension(:), allocatable :: iglob,locval
  logical, dimension(:), allocatable :: ifseg
  double precision, dimension(:), allocatable :: xp,yp,zp

  integer nglob,NGLOB_AB
  integer ieoff,ilocnum

! mass matrix
!  real(kind=CUSTOM_REAL), dimension(:), allocatable :: rmass

! boundary parameters locator
  integer, dimension(:), allocatable :: ibelm_xmin,ibelm_xmax,ibelm_ymin,ibelm_ymax,ibelm_bottom,ibelm_top

! ---- Moho Vars here ------
! ! Moho boundary locator
!   integer, dimension(:), allocatable :: ibelm_moho_top, ibelm_moho_bot
!   logical, dimension(:), allocatable :: is_moho_top, is_moho_bot

! ! 2-D jacobian and normals
!   real(kind=CUSTOM_REAL), dimension(:,:,:), allocatable :: jacobian2D_moho
!   real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: normal_moho

! ! number of elements on the boundaries
!   integer nspec_moho_top, nspec_moho_bottom
! ---------------------------

! 2-D jacobians and normals
!   real(kind=CUSTOM_REAL), dimension(:,:,:), allocatable :: &
!     jacobian2D_xmin,jacobian2D_xmax, &
!     jacobian2D_ymin,jacobian2D_ymax,jacobian2D_bottom,jacobian2D_top

!   real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: &
!     normal_xmin,normal_xmax,normal_ymin,normal_ymax,normal_bottom,normal_top

! MPI cut-planes parameters along xi and along eta
  logical, dimension(:,:), allocatable :: iMPIcut_xi,iMPIcut_eta

! name of the database file
  character(len=150) prname

! number of elements on the boundaries
  integer nspec2D_xmin,nspec2D_xmax,nspec2D_ymin,nspec2D_ymax

  integer i,j,k,ia,ispec,itype_element
  integer iproc_xi,iproc_eta

!  double precision rho,vp,vs
!  double precision c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66

! for the Harvard 3-D basin model
  double precision vp_block_gocad_MR(0:NX_GOCAD_MR-1,0:NY_GOCAD_MR-1,0:NZ_GOCAD_MR-1)
  double precision vp_block_gocad_HR(0:NX_GOCAD_HR-1,0:NY_GOCAD_HR-1,0:NZ_GOCAD_HR-1)
  integer irecord,nrecord,i_vp
  character(len=150) BASIN_MODEL_3D_MEDIUM_RES_FILE,BASIN_MODEL_3D_HIGH_RES_FILE

! for the harvard 3D salton sea model
  real :: vp_st_gocad(GOCAD_ST_NU,GOCAD_ST_NV,GOCAD_ST_NW)
!  double precision :: umesh, vmesh, wmesh, vp_st, vs_st, rho_st

! for Hauksson's model
  double precision, dimension(NLAYERS_HAUKSSON,NGRID_NEW_HAUKSSON,NGRID_NEW_HAUKSSON) :: vp_hauksson,vs_hauksson
  integer ilayer
  character(len=150 ) HAUKSSON_REGIONAL_MODEL_FILE

! Stacey put back
! indices for Clayton-Engquist absorbing conditions
!   integer, dimension(:,:), allocatable :: nimin,nimax,njmin,njmax,nkmin_xi,nkmin_eta
!   real(kind=CUSTOM_REAL), dimension(:,:,:,:), allocatable :: rho_vp,rho_vs

! flag indicating whether point is in the sediments
!  logical point_is_in_sediments
  logical, dimension(:,:,:,:), allocatable :: flag_sediments
  logical, dimension(:), allocatable :: not_fully_in_bedrock


! **************

! create the name for the database of the current slide and region
  call create_name_database(prname,myrank,LOCAL_PATH)

! Gauss-Lobatto-Legendre points of integration
!   allocate(xigll(NGLLX))
!   allocate(yigll(NGLLY))
!   allocate(zigll(NGLLZ))

! Gauss-Lobatto-Legendre weights of integration
!   allocate(wxgll(NGLLX))
!   allocate(wygll(NGLLY))
!   allocate(wzgll(NGLLZ))

! 3D shape functions and their derivatives
  ! allocate(shape3D(NGNOD,NGLLX,NGLLY,NGLLZ))
!   allocate(dershape3D(NDIM,NGNOD,NGLLX,NGLLY,NGLLZ))

! 2D shape functions and their derivatives
!   allocate(shape2D_x(NGNOD2D,NGLLY,NGLLZ))
!   allocate(shape2D_y(NGNOD2D,NGLLX,NGLLZ))
!   allocate(shape2D_bottom(NGNOD2D,NGLLX,NGLLY))
!   allocate(shape2D_top(NGNOD2D,NGLLX,NGLLY))
!   allocate(dershape2D_x(NDIM2D,NGNOD2D,NGLLY,NGLLZ))
!   allocate(dershape2D_y(NDIM2D,NGNOD2D,NGLLX,NGLLZ))
!   allocate(dershape2D_bottom(NDIM2D,NGNOD2D,NGLLX,NGLLY))
!   allocate(dershape2D_top(NDIM2D,NGNOD2D,NGLLX,NGLLY))

! array with model density
!   allocate(rhostore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(kappastore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(mustore(NGLLX,NGLLY,NGLLZ,nspec))

! array with anisotropy
!   allocate(c11store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c12store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c13store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c14store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c15store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c16store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c22store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c23store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c24store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c25store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c26store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c33store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c34store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c35store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c36store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c44store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c45store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c46store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c55store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c56store(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(c66store(NGLLX,NGLLY,NGLLZ,nspec))

! Stacey
!   allocate(rho_vp(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(rho_vs(NGLLX,NGLLY,NGLLZ,nspec))

! flag indicating whether point is in the sediments
  allocate(flag_sediments(NGLLX,NGLLY,NGLLZ,nspec))
  allocate(not_fully_in_bedrock(nspec))

! boundary locator
  allocate(iboun(6,nspec))

! arrays with mesh parameters
!   allocate(xixstore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(xiystore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(xizstore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(etaxstore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(etaystore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(etazstore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(gammaxstore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(gammaystore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(gammazstore(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(jacobianstore(NGLLX,NGLLY,NGLLZ,nspec))

! boundary parameters locator
  allocate(ibelm_xmin(NSPEC2DMAX_XMIN_XMAX))
  allocate(ibelm_xmax(NSPEC2DMAX_XMIN_XMAX))
  allocate(ibelm_ymin(NSPEC2DMAX_YMIN_YMAX))
  allocate(ibelm_ymax(NSPEC2DMAX_YMIN_YMAX))
  allocate(ibelm_bottom(NSPEC2D_BOTTOM))
  allocate(ibelm_top(NSPEC2D_TOP))

! 2-D jacobians and normals
!   allocate(jacobian2D_xmin(NGLLY,NGLLZ,NSPEC2DMAX_XMIN_XMAX))
!   allocate(jacobian2D_xmax(NGLLY,NGLLZ,NSPEC2DMAX_XMIN_XMAX))
!   allocate(jacobian2D_ymin(NGLLX,NGLLZ,NSPEC2DMAX_YMIN_YMAX))
!   allocate(jacobian2D_ymax(NGLLX,NGLLZ,NSPEC2DMAX_YMIN_YMAX))
!   allocate(jacobian2D_bottom(NGLLX,NGLLY,NSPEC2D_BOTTOM))
!   allocate(jacobian2D_top(NGLLX,NGLLY,NSPEC2D_TOP))

!   allocate(normal_xmin(NDIM,NGLLY,NGLLZ,NSPEC2DMAX_XMIN_XMAX))
!   allocate(normal_xmax(NDIM,NGLLY,NGLLZ,NSPEC2DMAX_XMIN_XMAX))
!   allocate(normal_ymin(NDIM,NGLLX,NGLLZ,NSPEC2DMAX_YMIN_YMAX))
!   allocate(normal_ymax(NDIM,NGLLX,NGLLZ,NSPEC2DMAX_YMIN_YMAX))
!   allocate(normal_bottom(NDIM,NGLLX,NGLLY,NSPEC2D_BOTTOM))
!   allocate(normal_top(NDIM,NGLLX,NGLLY,NSPEC2D_TOP))

! Moho boundary parameters, 2-D jacobians and normals
!   if (SAVE_MOHO_MESH) then
!     allocate(ibelm_moho_top(NSPEC2D_BOTTOM))
!     allocate(ibelm_moho_bot(NSPEC2D_BOTTOM))
!     allocate(is_moho_top(nspec))
!     allocate(is_moho_bot(nspec))
!     is_moho_top = .false.
!     is_moho_bot = .false.
!     nspec_moho_top = 0
!     nspec_moho_bottom = 0
!     allocate(jacobian2D_moho(NGLLX,NGLLY,NSPEC2D_BOTTOM))
!     allocate(normal_moho(NDIM,NGLLX,NGLLY,NSPEC2D_BOTTOM))
!   endif


! Stacey put back
!   allocate(nimin(2,NSPEC2DMAX_YMIN_YMAX))
!   allocate(nimax(2,NSPEC2DMAX_YMIN_YMAX))
!   allocate(njmin(2,NSPEC2DMAX_XMIN_XMAX))
!   allocate(njmax(2,NSPEC2DMAX_XMIN_XMAX))
!   allocate(nkmin_xi(2,NSPEC2DMAX_XMIN_XMAX))
!   allocate(nkmin_eta(2,NSPEC2DMAX_YMIN_YMAX))

! MPI cut-planes parameters along xi and along eta
  allocate(iMPIcut_xi(2,nspec))
  allocate(iMPIcut_eta(2,nspec))

! set up coordinates of the Gauss-Lobatto-Legendre points
!   call zwgljd(xigll,wxgll,NGLLX,GAUSSALPHA,GAUSSBETA)
!   call zwgljd(yigll,wygll,NGLLY,GAUSSALPHA,GAUSSBETA)
!   call zwgljd(zigll,wzgll,NGLLZ,GAUSSALPHA,GAUSSBETA)

! if number of points is odd, the middle abscissa is exactly zero
!   if(mod(NGLLX,2) /= 0) xigll((NGLLX-1)/2+1) = ZERO
!   if(mod(NGLLY,2) /= 0) yigll((NGLLY-1)/2+1) = ZERO
!   if(mod(NGLLZ,2) /= 0) zigll((NGLLZ-1)/2+1) = ZERO

! get the 3-D shape functions
!  call get_shape3D(myrank,shape3D,dershape3D,xigll,yigll,zigll)

! get the 2-D shape functions
!   call get_shape2D(myrank,shape2D_x,dershape2D_x,yigll,zigll,NGLLY,NGLLZ)
!   call get_shape2D(myrank,shape2D_y,dershape2D_y,xigll,zigll,NGLLX,NGLLZ)
!   call get_shape2D(myrank,shape2D_bottom,dershape2D_bottom,xigll,yigll,NGLLX,NGLLY)
!   call get_shape2D(myrank,shape2D_top,dershape2D_top,xigll,yigll,NGLLX,NGLLY)

! allocate memory for arrays
  allocate(iglob(npointot))
  allocate(locval(npointot))
  allocate(ifseg(npointot))
  allocate(xp(npointot))
  allocate(yp(npointot))
  allocate(zp(npointot))

!--- read Hauksson's model
  if(HAUKSSON_REGIONAL_MODEL) then
    call get_value_string(HAUKSSON_REGIONAL_MODEL_FILE, &
                          'model.HAUKSSON_REGIONAL_MODEL_FILE', &
                          'DATA/hauksson_model/hauksson_final_grid_smooth.dat')
!    call get_value_string(HAUKSSON_REGIONAL_MODEL_FILE, &
!                          'model.HAUKSSON_REGIONAL_MODEL_FILE', &
!                          'DATA/lin_model/lin_final_grid_smooth.dat')
    open(unit=14,file=HAUKSSON_REGIONAL_MODEL_FILE,status='old',action='read')
    do iy = 1,NGRID_NEW_HAUKSSON
      do ix = 1,NGRID_NEW_HAUKSSON
        read(14,*) (vp_hauksson(ilayer,ix,iy),ilayer=1,NLAYERS_HAUKSSON), &
                   (vs_hauksson(ilayer,ix,iy),ilayer=1,NLAYERS_HAUKSSON)
      enddo
    enddo
    close(14)
    vp_hauksson(:,:,:) = vp_hauksson(:,:,:) * 1000.d0
    vs_hauksson(:,:,:) = vs_hauksson(:,:,:) * 1000.d0
  endif

!--- read the Harvard 3-D basin model
  if(HARVARD_3D_GOCAD_MODEL) then

! read medium-resolution model

! initialize array to undefined values everywhere
  vp_block_gocad_MR(:,:,:) = 20000.

! read Vp from extracted text file
  call get_value_string(BASIN_MODEL_3D_MEDIUM_RES_FILE, &
                        'model.BASIN_MODEL_3D_MEDIUM_RES_FILE', &
                        'DATA/la_3D_block_harvard/la_3D_medium_res/LA_MR_voxet_extracted.txt')
  open(unit=27,file=BASIN_MODEL_3D_MEDIUM_RES_FILE,status='old',action='read')
  read(27,*) nrecord
  do irecord = 1,nrecord
    read(27,*) ix,iy,iz,i_vp
    if(ix<0 .or. ix>NX_GOCAD_MR-1 .or. iy<0 .or. iy>NY_GOCAD_MR-1 .or. iz<0 .or. iz>NZ_GOCAD_MR-1) &
      stop 'wrong array index read in Gocad medium-resolution file'
    vp_block_gocad_MR(ix,iy,iz) = dble(i_vp)
  enddo
  close(27)

! read high-resolution model

! initialize array to undefined values everywhere
  vp_block_gocad_HR(:,:,:) = 20000.

! read Vp from extracted text file
  call get_value_string(BASIN_MODEL_3D_HIGH_RES_FILE, &
                        'model.BASIN_MODEL_3D_HIGH_RES_FILE', &
                        'DATA/la_3D_block_harvard/la_3D_high_res/LA_HR_voxet_extracted.txt')
  open(unit=27,file=BASIN_MODEL_3D_HIGH_RES_FILE,status='old',action='read')
  read(27,*) nrecord
  do irecord = 1,nrecord
    read(27,*) ix,iy,iz,i_vp
    if(ix<0 .or. ix>NX_GOCAD_HR-1 .or. iy<0 .or. iy>NY_GOCAD_HR-1 .or. iz<0 .or. iz>NZ_GOCAD_HR-1) &
      stop 'wrong array index read in Gocad high-resolution file'
    vp_block_gocad_HR(ix,iy,iz) = dble(i_vp)
  enddo
  close(27)

! read Salton Trough model
  call read_salton_sea_model(vp_st_gocad)

  endif

!--- apply heuristic rule to modify doubling regions to balance angles

  if(APPLY_HEURISTIC_RULE .and. .not. USE_REGULAR_MESH) then

! define number of subregions affected by heuristic rule in doubling regions
  nsubregions = 8

  do isubregion = 1,nsubregions

! define shape of elements for heuristic
    call define_subregions_heuristic(myrank,isubregion,iaddx,iaddy,iaddz, &
              ix1,ix2,dix,iy1,iy2,diy,ir1,ir2,dir,iax,iay,iar, &
              itype_element,npx,npy, &
              NER_BOTTOM_MOHO,NER_MOHO_16,NER_16_BASEMENT,NER_BASEMENT_SEDIM)

! loop on all the mesh points in current subregion
  do ir = ir1,ir2,dir
    do iy = iy1,iy2,diy
      do ix = ix1,ix2,dix

! this heuristic rule is only valid for 8-node elements
! it would not work in the case of 27 nodes

!----
    if(itype_element == ITYPE_UNUSUAL_1) then

! side 1
      horiz_size = xgrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2)) &
                 - xgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      xgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) = &
         xgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + horiz_size * MAGIC_RATIO

      vert_size = zgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) &
                 - zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      zgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) = &
         zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + vert_size * MAGIC_RATIO / 0.50

! side 2
      horiz_size = xgrid(ir+iar*iaddz(3),ix+iax*iaddx(3),iy+iay*iaddy(3)) &
                 - xgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4))
      xgrid(ir+iar*iaddz(8),ix+iax*iaddx(8),iy+iay*iaddy(8)) = &
         xgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4)) + horiz_size * MAGIC_RATIO

      vert_size = zgrid(ir+iar*iaddz(8),ix+iax*iaddx(8),iy+iay*iaddy(8)) &
                 - zgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4))
      zgrid(ir+iar*iaddz(8),ix+iax*iaddx(8),iy+iay*iaddy(8)) = &
         zgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4)) + vert_size * MAGIC_RATIO / 0.50

!----
    else if(itype_element == ITYPE_UNUSUAL_1p) then

! side 1
      horiz_size = xgrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2)) &
                 - xgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      xgrid(ir+iar*iaddz(6),ix+iax*iaddx(6),iy+iay*iaddy(6)) = &
         xgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + horiz_size * (1. - MAGIC_RATIO)

      vert_size = zgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) &
                 - zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      zgrid(ir+iar*iaddz(6),ix+iax*iaddx(6),iy+iay*iaddy(6)) = &
         zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + vert_size * MAGIC_RATIO / 0.50

! side 2
      horiz_size = xgrid(ir+iar*iaddz(3),ix+iax*iaddx(3),iy+iay*iaddy(3)) &
                 - xgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4))
      xgrid(ir+iar*iaddz(7),ix+iax*iaddx(7),iy+iay*iaddy(7)) = &
         xgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4)) + horiz_size * (1. - MAGIC_RATIO)

      vert_size = zgrid(ir+iar*iaddz(8),ix+iax*iaddx(8),iy+iay*iaddy(8)) &
                 - zgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4))
      zgrid(ir+iar*iaddz(7),ix+iax*iaddx(7),iy+iay*iaddy(7)) = &
         zgrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4)) + vert_size * MAGIC_RATIO / 0.50

!----
    else if(itype_element == ITYPE_UNUSUAL_4) then

! side 1
      horiz_size = ygrid(ir+iar*iaddz(3),ix+iax*iaddx(3),iy+iay*iaddy(3)) &
                 - ygrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2))
      ygrid(ir+iar*iaddz(7),ix+iax*iaddx(7),iy+iay*iaddy(7)) = &
         ygrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2)) + horiz_size * (1. - MAGIC_RATIO)

      vert_size = zgrid(ir+iar*iaddz(6),ix+iax*iaddx(6),iy+iay*iaddy(6)) &
                 - zgrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2))
      zgrid(ir+iar*iaddz(7),ix+iax*iaddx(7),iy+iay*iaddy(7)) = &
         zgrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2)) + vert_size * MAGIC_RATIO / 0.50

! side 2
      horiz_size = ygrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4)) &
                 - ygrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      ygrid(ir+iar*iaddz(8),ix+iax*iaddx(8),iy+iay*iaddy(8)) = &
         ygrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + horiz_size * (1. - MAGIC_RATIO)

      vert_size = zgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) &
                 - zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      zgrid(ir+iar*iaddz(8),ix+iax*iaddx(8),iy+iay*iaddy(8)) = &
         zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + vert_size * MAGIC_RATIO / 0.50

!----
    else if(itype_element == ITYPE_UNUSUAL_4p) then

! side 1
      horiz_size = ygrid(ir+iar*iaddz(3),ix+iax*iaddx(3),iy+iay*iaddy(3)) &
                 - ygrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2))
      ygrid(ir+iar*iaddz(6),ix+iax*iaddx(6),iy+iay*iaddy(6)) = &
         ygrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2)) + horiz_size * MAGIC_RATIO

      vert_size = zgrid(ir+iar*iaddz(6),ix+iax*iaddx(6),iy+iay*iaddy(6)) &
                 - zgrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2))
      zgrid(ir+iar*iaddz(6),ix+iax*iaddx(6),iy+iay*iaddy(6)) = &
         zgrid(ir+iar*iaddz(2),ix+iax*iaddx(2),iy+iay*iaddy(2)) + vert_size * MAGIC_RATIO / 0.50

! side 2
      horiz_size = ygrid(ir+iar*iaddz(4),ix+iax*iaddx(4),iy+iay*iaddy(4)) &
                 - ygrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      ygrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) = &
         ygrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + horiz_size * MAGIC_RATIO

      vert_size = zgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) &
                 - zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1))
      zgrid(ir+iar*iaddz(5),ix+iax*iaddx(5),iy+iay*iaddy(5)) = &
         zgrid(ir+iar*iaddz(1),ix+iax*iaddx(1),iy+iay*iaddy(1)) + vert_size * MAGIC_RATIO / 0.50

    endif

      enddo
    enddo
  enddo

  enddo

  endif

!---

! generate the elements in all the regions of the mesh
  ispec = 0

! define number of subregions in the mesh
  if(USE_REGULAR_MESH) then
    nsubregions = 2
  else
    if(NER_SEDIM > 1) then
      nsubregions = 30
    else
      nsubregions = 29
    endif
  endif

  do isubregion = 1,nsubregions

! define shape of elements
    call define_subregions(myrank,isubregion,iaddx,iaddy,iaddz, &
              ix1,ix2,dix,iy1,iy2,diy,ir1,ir2,dir,iax,iay,iar, &
              doubling_index,npx,npy, &
              NER_BOTTOM_MOHO,NER_MOHO_16,NER_16_BASEMENT,NER_BASEMENT_SEDIM,NER_SEDIM,NER,USE_REGULAR_MESH)

! loop on all the mesh points in current subregion
  do ir = ir1,ir2,dir
    do iy = iy1,iy2,diy
      do ix = ix1,ix2,dix

!       loop over the NGNOD nodes
        do ia=1,NGNOD
          xelm(ia) = xgrid(ir+iar*iaddz(ia),ix+iax*iaddx(ia),iy+iay*iaddy(ia))
          yelm(ia) = ygrid(ir+iar*iaddz(ia),ix+iax*iaddx(ia),iy+iay*iaddy(ia))
          zelm(ia) = zgrid(ir+iar*iaddz(ia),ix+iax*iaddx(ia),iy+iay*iaddy(ia))
        enddo

! add one spectral element to the list and store its material number
        ispec = ispec + 1
        if(ispec > nspec) call exit_MPI(myrank,'ispec greater than nspec in mesh creation')
        idoubling(ispec) = doubling_index







! ! assign Moho surface element
!         if (SAVE_MOHO_MESH) then
!         if (isubregion == 15 .and. ir == ir1) then
!           nspec_moho_top = nspec_moho_top + 1
!           if (nspec_moho_top > NSPEC2D_BOTTOM) call exit_mpi(myrank,"Error counting moho top elements")
!           ibelm_moho_top(nspec_moho_top) = ispec
!           call compute_jacobian_2D(myrank,nspec_moho_top,xelm(1:NGNOD2D),yelm(1:NGNOD2D),zelm(1:NGNOD2D), &
!                      dershape2D_bottom,jacobian2D_moho,normal_moho,NGLLX,NGLLY,NSPEC2D_BOTTOM)
!           is_moho_top(ispec) = .true.
!         else if (isubregion == 28 .and. ir+dir > ir2) then
!           nspec_moho_bottom = nspec_moho_bottom + 1
!           if (nspec_moho_bottom > NSPEC2D_BOTTOM) call exit_mpi(myrank,"Error counting moho bottom elements")
!           ibelm_moho_bot(nspec_moho_bottom) = ispec
!           is_moho_bot(ispec) = .true.
!         endif
!         endif

! ! initialize flag indicating whether element is in sediments
!   not_fully_in_bedrock(ispec) = .false.

! ! create mesh element
!   do k=1,NGLLZ
!     do j=1,NGLLY
!       do i=1,NGLLX

! ! compute mesh coordinates
!        xmesh = ZERO
!        ymesh = ZERO
!        zmesh = ZERO
!        do ia=1,NGNOD
!          xmesh = xmesh + shape3D(ia,i,j,k)*xelm(ia)
!          ymesh = ymesh + shape3D(ia,i,j,k)*yelm(ia)
!          zmesh = zmesh + shape3D(ia,i,j,k)*zelm(ia)
!        enddo

!        xstore_local(i,j,k) = xmesh
!        ystore_local(i,j,k) = ymesh
!        zstore_local(i,j,k) = zmesh

! ! initialize flag indicating whether point is in the sediments
!        point_is_in_sediments = .false.

!        if(ANISOTROPY) then
!           call aniso_model(doubling_index,zmesh,rho,vp,vs,c11,c12,c13,c14,c15,c16,&
!                c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66)
!        else
! ! get the regional model parameters
!           if(HAUKSSON_REGIONAL_MODEL) then
! ! get density from socal model
!              call socal_model(doubling_index,rho,vp,vs)
! ! get vp and vs from Hauksson
!              call hauksson_model(vp_hauksson,vs_hauksson,xmesh,ymesh,zmesh,vp,vs,MOHO_MAP_LUPEI)
! ! if Moho map is used, then assume homogeneous medium below the Moho
! ! and use bottom layer of Hauksson's model in the halfspace
!              if(MOHO_MAP_LUPEI .and. doubling_index == IFLAG_HALFSPACE_MOHO) &
!                   call socal_model(IFLAG_HALFSPACE_MOHO,rho,vp,vs)
!           else
!              call socal_model(doubling_index,rho,vp,vs)
! ! include attenuation in first SoCal layer if needed
! ! uncomment line below to include attenuation in the 1D case
! !        if(zmesh >= DEPTH_5p5km_SOCAL) point_is_in_sediments = .true.
!           endif

! ! get the Harvard 3-D basin model
!           if(HARVARD_3D_GOCAD_MODEL .and. &
!                (doubling_index == IFLAG_ONE_LAYER_TOPOGRAPHY &
!                .or. doubling_index == IFLAG_BASEMENT_TOPO) &
!                .and. xmesh >= ORIG_X_GOCAD_MR &
!                .and. xmesh <= END_X_GOCAD_MR &
!                .and. ymesh >= ORIG_Y_GOCAD_MR &
!                .and. ymesh <= END_Y_GOCAD_MR) then

! ! use medium-resolution model first
!              call interpolate_gocad_block_MR(vp_block_gocad_MR, &
!                   xmesh,ymesh,zmesh,rho,vp,vs,point_is_in_sediments, &
!                   VP_MIN_GOCAD,VP_VS_RATIO_GOCAD_TOP,VP_VS_RATIO_GOCAD_BOTTOM, &
!                   IMPOSE_MINIMUM_VP_GOCAD,THICKNESS_TAPER_BLOCK_MR, &
!                   vp_hauksson,vs_hauksson,doubling_index,HAUKSSON_REGIONAL_MODEL,&
!                   MOHO_MAP_LUPEI)

! ! then superimpose high-resolution model
!              if(xmesh >= ORIG_X_GOCAD_HR &
!                   .and. xmesh <= END_X_GOCAD_HR &
!                   .and. ymesh >= ORIG_Y_GOCAD_HR &
!                   .and. ymesh <= END_Y_GOCAD_HR) &
!                   call interpolate_gocad_block_HR(vp_block_gocad_HR,vp_block_gocad_MR,&
!                   xmesh,ymesh,zmesh,rho,vp,vs,point_is_in_sediments, &
!                   VP_MIN_GOCAD,VP_VS_RATIO_GOCAD_TOP,VP_VS_RATIO_GOCAD_BOTTOM, &
!                   IMPOSE_MINIMUM_VP_GOCAD,THICKNESS_TAPER_BLOCK_HR, &
!                   vp_hauksson,vs_hauksson,doubling_index,HAUKSSON_REGIONAL_MODEL, &
!                   MOHO_MAP_LUPEI)

!           endif
! ! get the Harvard Salton Trough model
!           if (HARVARD_3D_GOCAD_MODEL) then
!             call vx_xyz2uvw(xmesh, ymesh, zmesh, umesh, vmesh, wmesh)
!             if (umesh >= 0 .and. umesh <= GOCAD_ST_NU-1 .and. &
!                   vmesh >= 0 .and. vmesh <=  GOCAD_ST_NV-1 .and. &
!                   wmesh >= 0 .and. wmesh <= GOCAD_ST_NW-1) then
!               call vx_xyz_interp(umesh,vmesh,wmesh, vp_st, vs_st, rho_st, vp_st_gocad)
!               if (abs(vp_st - GOCAD_ST_NO_DATA_VALUE) > 1.0d-3) then
!                 vp = vp_st
!                 vs = vs_st
!                 rho = rho_st
!               endif
!             endif
!           endif
!        endif
! ! store flag indicating whether point is in the sediments
!   flag_sediments(i,j,k,ispec) = point_is_in_sediments
!   if(point_is_in_sediments) not_fully_in_bedrock(ispec) = .true.

! ! define elastic parameters in the model
! ! distinguish between single and double precision for reals
!   if(ANISOTROPY) then

!        if(CUSTOM_REAL == SIZE_REAL) then
!          rhostore(i,j,k,ispec) = sngl(rho)
!          kappastore(i,j,k,ispec) = sngl(rho*(vp*vp - 4.d0*vs*vs/3.d0))
!          mustore(i,j,k,ispec) = sngl(rho*vs*vs)
!          c11store(i,j,k,ispec) = sngl(c11)
!          c12store(i,j,k,ispec) = sngl(c12)
!          c13store(i,j,k,ispec) = sngl(c13)
!          c14store(i,j,k,ispec) = sngl(c14)
!          c15store(i,j,k,ispec) = sngl(c15)
!          c16store(i,j,k,ispec) = sngl(c16)
!          c22store(i,j,k,ispec) = sngl(c22)
!          c23store(i,j,k,ispec) = sngl(c23)
!          c24store(i,j,k,ispec) = sngl(c24)
!          c25store(i,j,k,ispec) = sngl(c25)
!          c26store(i,j,k,ispec) = sngl(c26)
!          c33store(i,j,k,ispec) = sngl(c33)
!          c34store(i,j,k,ispec) = sngl(c34)
!          c35store(i,j,k,ispec) = sngl(c35)
!          c36store(i,j,k,ispec) = sngl(c36)
!          c44store(i,j,k,ispec) = sngl(c44)
!          c45store(i,j,k,ispec) = sngl(c45)
!          c46store(i,j,k,ispec) = sngl(c46)
!          c55store(i,j,k,ispec) = sngl(c55)
!          c56store(i,j,k,ispec) = sngl(c56)
!          c66store(i,j,k,ispec) = sngl(c66)
! ! Stacey
!          rho_vp(i,j,k,ispec) = sngl(rho*vp)
!          rho_vs(i,j,k,ispec) = sngl(rho*vs)
!       else
!          rhostore(i,j,k,ispec) = rho
!          kappastore(i,j,k,ispec) = rho*(vp*vp - 4.d0*vs*vs/3.d0)
!          mustore(i,j,k,ispec) = rho*vs*vs
!          c11store(i,j,k,ispec) = c11
!          c12store(i,j,k,ispec) = c12
!          c13store(i,j,k,ispec) = c13
!          c14store(i,j,k,ispec) = c14
!          c15store(i,j,k,ispec) = c15
!          c16store(i,j,k,ispec) = c16
!          c22store(i,j,k,ispec) = c22
!          c23store(i,j,k,ispec) = c23
!          c24store(i,j,k,ispec) = c24
!          c25store(i,j,k,ispec) = c25
!          c26store(i,j,k,ispec) = c26
!          c33store(i,j,k,ispec) = c33
!          c34store(i,j,k,ispec) = c34
!          c35store(i,j,k,ispec) = c35
!          c36store(i,j,k,ispec) = c36
!          c44store(i,j,k,ispec) = c44
!          c45store(i,j,k,ispec) = c45
!          c46store(i,j,k,ispec) = c46
!          c55store(i,j,k,ispec) = c55
!          c56store(i,j,k,ispec) = c56
!          c66store(i,j,k,ispec) = c66
! ! Stacey
!          rho_vp(i,j,k,ispec) = rho*vp
!          rho_vs(i,j,k,ispec) = rho*vs
!       endif


!    else
!       if(CUSTOM_REAL == SIZE_REAL) then
!          rhostore(i,j,k,ispec) = sngl(rho)
!          kappastore(i,j,k,ispec) = sngl(rho*(vp*vp - 4.d0*vs*vs/3.d0))
!          mustore(i,j,k,ispec) = sngl(rho*vs*vs)

! ! Stacey
!          rho_vp(i,j,k,ispec) = sngl(rho*vp)
!          rho_vs(i,j,k,ispec) = sngl(rho*vs)
!       else
!          rhostore(i,j,k,ispec) = rho
!          kappastore(i,j,k,ispec) = rho*(vp*vp - 4.d0*vs*vs/3.d0)
!          mustore(i,j,k,ispec) = rho*vs*vs

! ! Stacey
!          rho_vp(i,j,k,ispec) = rho*vp
!          rho_vs(i,j,k,ispec) = rho*vs
!       endif
!    endif

! enddo
! enddo
! enddo

! ! compute coordinates and jacobian
!         call calc_jacobian(myrank,xixstore,xiystore,xizstore, &
!                etaxstore,etaystore,etazstore, &
!                gammaxstore,gammaystore,gammazstore,jacobianstore, &
!                xstore,ystore,zstore, &
!                xelm,yelm,zelm,shape3D,dershape3D,ispec,nspec)

! store coordinates 
        call store_coords(xstore,ystore,zstore,xelm,yelm,zelm,ispec,nspec)

! detect mesh boundaries
        call get_flags_boundaries(nspec,iproc_xi,iproc_eta,ispec,doubling_index, &
             xstore(:,:,:,ispec),ystore(:,:,:,ispec),zstore(:,:,:,ispec), &
             iboun,iMPIcut_xi,iMPIcut_eta,NPROC_XI,NPROC_ETA, &
             UTM_X_MIN,UTM_X_MAX,UTM_Y_MIN,UTM_Y_MAX,Z_DEPTH_BLOCK)

! end of loop on all the mesh points in current subregion
      enddo
    enddo
  enddo

! end of loop on all the subregions of the current region the mesh
  enddo






! check total number of spectral elements created
  if(ispec /= nspec) call exit_MPI(myrank,'ispec should equal nspec')
!   if (SAVE_MOHO_MESH) then
!     if (nspec_moho_top /= NSPEC2D_BOTTOM .or. nspec_moho_bottom /= NSPEC2D_BOTTOM) &
!                call exit_mpi(myrank, "nspec_moho should equal NSPEC2D_BOTTOM")
!   endif


  do ispec=1,nspec
  ieoff = NGLLCUBE*(ispec-1)
  ilocnum = 0
  do k=1,NGLLZ
    do j=1,NGLLY
      do i=1,NGLLX
        ilocnum = ilocnum + 1
        xp(ilocnum+ieoff) = xstore(i,j,k,ispec)
        yp(ilocnum+ieoff) = ystore(i,j,k,ispec)
        zp(ilocnum+ieoff) = zstore(i,j,k,ispec)
      enddo
    enddo
  enddo
  enddo

  call get_global(nspec,xp,yp,zp,iglob,locval,ifseg,nglob,npointot,UTM_X_MIN,UTM_X_MAX)

  !PLL
  allocate(nodes_coords(nglob,3))

! put in classical format
  do ispec=1,nspec
  ieoff = NGLLCUBE*(ispec-1)
  ilocnum = 0
  do k=1,NGLLZ
    do j=1,NGLLY
      do i=1,NGLLX
        ilocnum = ilocnum + 1
        ibool(i,j,k,ispec) = iglob(ilocnum+ieoff)
        nodes_coords(iglob(ilocnum+ieoff),1) = xstore(i,j,k,ispec)
        nodes_coords(iglob(ilocnum+ieoff),2) = ystore(i,j,k,ispec)
        nodes_coords(iglob(ilocnum+ieoff),3) = zstore(i,j,k,ispec)
      enddo
    enddo
  enddo
  enddo

  if(minval(ibool(:,:,:,:)) /= 1 .or. maxval(ibool(:,:,:,:)) /= NGLOB_AB) &
    call exit_MPI(myrank,'incorrect global numbering')

! ! create a new indirect addressing array instead, to reduce cache misses
! ! in memory access in the solver
!   allocate(copy_ibool_ori(NGLLX,NGLLY,NGLLZ,nspec))
!   allocate(mask_ibool(nglob))
!   mask_ibool(:) = -1
!   copy_ibool_ori(:,:,:,:) = ibool(:,:,:,:)

!   inumber = 0
!   do ispec=1,nspec
!   do k=1,NGLLZ
!     do j=1,NGLLY
!       do i=1,NGLLX
!         if(mask_ibool(copy_ibool_ori(i,j,k,ispec)) == -1) then
! ! create a new point
!           inumber = inumber + 1
!           ibool(i,j,k,ispec) = inumber
!           mask_ibool(copy_ibool_ori(i,j,k,ispec)) = inumber
!         else
! ! use an existing point created previously
!           ibool(i,j,k,ispec) = mask_ibool(copy_ibool_ori(i,j,k,ispec))
!         endif
!       enddo
!     enddo
!   enddo
!   enddo
!   deallocate(copy_ibool_ori)
!   deallocate(mask_ibool)

! ! creating mass matrix (will be fully assembled with MPI in the solver)
!   allocate(rmass(nglob))
!   rmass(:) = 0._CUSTOM_REAL

!   do ispec=1,nspec
!   do k=1,NGLLZ
!     do j=1,NGLLY
!       do i=1,NGLLX
!         weight=wxgll(i)*wygll(j)*wzgll(k)
!         iglobnum=ibool(i,j,k,ispec)

!         jacobianl=jacobianstore(i,j,k,ispec)

! ! distinguish between single and double precision for reals
!     if(CUSTOM_REAL == SIZE_REAL) then
!       rmass(iglobnum) = rmass(iglobnum) + &
!              sngl(dble(rhostore(i,j,k,ispec)) * dble(jacobianl) * weight)
!     else
!       rmass(iglobnum) = rmass(iglobnum) + rhostore(i,j,k,ispec) * jacobianl * weight
!     endif

!       enddo
!     enddo
!   enddo
!   enddo

!   call get_jacobian_boundaries(myrank,iboun,nspec,xstore,ystore,zstore, &
!       dershape2D_x,dershape2D_y,dershape2D_bottom,dershape2D_top, &
!       ibelm_xmin,ibelm_xmax,ibelm_ymin,ibelm_ymax,ibelm_bottom,ibelm_top, &
!       nspec2D_xmin,nspec2D_xmax,nspec2D_ymin,nspec2D_ymax, &
!               jacobian2D_xmin,jacobian2D_xmax, &
!               jacobian2D_ymin,jacobian2D_ymax, &
!               jacobian2D_bottom,jacobian2D_top, &
!               normal_xmin,normal_xmax, &
!               normal_ymin,normal_ymax, &
!               normal_bottom,normal_top, &
!               NSPEC2D_BOTTOM,NSPEC2D_TOP, &
!               NSPEC2DMAX_XMIN_XMAX,NSPEC2DMAX_YMIN_YMAX)

  call store_boundaries(myrank,iboun,nspec, &
      ibelm_xmin,ibelm_xmax,ibelm_ymin,ibelm_ymax,ibelm_bottom,ibelm_top, &
      nspec2D_xmin,nspec2D_xmax,nspec2D_ymin,nspec2D_ymax, &
      NSPEC2D_BOTTOM,NSPEC2D_TOP, &
      NSPEC2DMAX_XMIN_XMAX,NSPEC2DMAX_YMIN_YMAX)

! create MPI buffers
! arrays locval(npointot) and ifseg(npointot) used to save memory
  call get_MPI_cutplanes_xi(myrank,prname,nspec,iMPIcut_xi,ibool, &
                  xstore,ystore,zstore,ifseg,npointot, &
                  NSPEC2D_A_ETA,NSPEC2D_B_ETA)
  call get_MPI_cutplanes_eta(myrank,prname,nspec,iMPIcut_eta,ibool, &
                  xstore,ystore,zstore,ifseg,npointot, &
                  NSPEC2D_A_XI,NSPEC2D_B_XI)

  call save_databases(prname,nspec,nglob,iproc_xi,iproc_eta,NPROC_XI,NPROC_ETA,addressing,iMPIcut_xi,iMPIcut_eta,&
       ibool,nodes_coords,idoubling,nspec2D_xmin,nspec2D_xmax,nspec2D_ymin,nspec2D_ymax,NSPEC2D_BOTTOM,&
       NSPEC2DMAX_XMIN_XMAX,NSPEC2DMAX_YMIN_YMAX,ibelm_xmin,ibelm_xmax,ibelm_ymin,ibelm_ymax,ibelm_bottom)



  end subroutine create_regions_mesh
