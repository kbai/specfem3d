(cd obj; mkdir -p dec)
(cd obj; mkdir -p mesh)
(cd obj; mkdir -p gen)
(cd obj; mkdir -p spec)
make -C src/decompose_mesh
make -C src/meshfem3D
make -C src/generate_databases generate_databases
make -C src/specfem3D specfem3D
make[1]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh'
echo "Using bundled Scotch"
Using bundled Scotch
make[1]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/meshfem3D'
mpiifort -c -o ../../obj/mesh/adios_manager_stubs.shared_noadios.o ../shared//adios_manager_stubs.f90
make[1]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/generate_databases'
mpiifort -c -o ../../obj/gen/adios_manager_stubs.shared_noadios.o ../shared//adios_manager_stubs.f90
make -C scotch/src
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/ -o  ../../bin/xdecompose_mesh ../../obj/dec/part_decompose_mesh.o ../../obj/dec/decompose_mesh.o ../../obj/dec/fault_scotch.o ../../obj/dec/get_value_parameters.shared.o ../../obj/dec/param_reader.cc.o ../../obj/dec/read_parameter_file.shared.o ../../obj/dec/read_value_parameters.shared.o ../../obj/dec/program_decompose_mesh.o -I"scotch/include"  -L"scotch/lib" -lscotch -lscotcherr
mpiifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/mesh/safe_alloc_mod.shared.o ../shared/safe_alloc_mod.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/assemble_MPI_scalar.shared.o ../shared/assemble_MPI_scalar.f90
make[1]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/specfem3D'
mpiifort -c -o ../../obj/spec/adios_manager_stubs.shared_noadios.o ../shared//adios_manager_stubs.f90
make[2]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src'
(cd libscotch ;      make VERSION=5 RELEASE=1 PATCHLEVEL=12 scotch && make install)
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/check_mesh_resolution.shared.o ../shared/check_mesh_resolution.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/create_name_database.shared.o ../shared/create_name_database.f90
make[3]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotch'
make CC="gcc" CCD="gcc"	\
					scotch.h				\
					scotchf.h				\
					libscotch.a				\
					libscotcherr.a			\
					libscotcherrexit.a
make[4]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotch'
make[4]: `scotch.h' is up to date.
make[4]: `scotchf.h' is up to date.
make[4]: `libscotch.a' is up to date.
make[4]: `libscotcherr.a' is up to date.
make[4]: `libscotcherrexit.a' is up to date.
make[4]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotch'
make[3]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotch'
make[3]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotch'
mpiifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/mesh/read_parameter_file.shared.o ../shared/read_parameter_file.f90
make[3]: Nothing to be done for `install'.
make[3]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotch'
(cd scotch ;         make VERSION=5 RELEASE=1 PATCHLEVEL=12 scotch && make install)
make[3]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/scotch'
make CC="gcc" SCOTCHLIB=scotch	\
					acpl				\
					amk_ccc				\
					amk_fft2				\
					amk_grf				\
					amk_hy				\
					amk_m2				\
					amk_p2				\
					atst				\
					gbase				\
					gcv				\
					gmap				\
					gmk_hy				\
					gmk_m2				\
					gmk_m3				\
					gmk_msh				\
					gmk_ub2				\
					gmtst				\
					gord				\
					gotst				\
					gout				\
					gpart				\
					gscat				\
					gtst				\
					mcv				\
					mmk_m2				\
					mmk_m3				\
					mord				\
					mtst
make[4]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/scotch'
mpiifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/mesh/read_value_parameters.shared.o ../shared/read_value_parameters.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage  -I../shared/ -c -o ../../obj/spec/fault_solver_common.o fault_solver_common.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage  -I../shared/ -c -o ../../obj/spec/fault_solver_dynamic.o fault_solver_dynamic.f90
make[4]: `acpl' is up to date.
make[4]: `amk_ccc' is up to date.
make[4]: `amk_fft2' is up to date.
make[4]: `amk_grf' is up to date.
make[4]: `amk_hy' is up to date.
make[4]: `amk_m2' is up to date.
make[4]: `amk_p2' is up to date.
make[4]: `atst' is up to date.
make[4]: `gbase' is up to date.
make[4]: `gcv' is up to date.
make[4]: `gmap' is up to date.
make[4]: `gmk_hy' is up to date.
make[4]: `gmk_m2' is up to date.
make[4]: `gmk_m3' is up to date.
make[4]: `gmk_msh' is up to date.
make[4]: `gmk_ub2' is up to date.
make[4]: `gmtst' is up to date.
make[4]: `gord' is up to date.
make[4]: `gotst' is up to date.
make[4]: `gout' is up to date.
make[4]: `gpart' is up to date.
make[4]: `gscat' is up to date.
make[4]: `gtst' is up to date.
make[4]: `mcv' is up to date.
make[4]: `mmk_m2' is up to date.
make[4]: `mmk_m3' is up to date.
make[4]: `mord' is up to date.
make[4]: `mtst' is up to date.
make[4]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/scotch'
make[3]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/scotch'
make[3]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/scotch'
cp dggath dgmap dgord dgpart dgscat dgtst ../../bin
make[3]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/scotch'
(cd libscotchmetis ; make                                                                scotch && make install)
make[3]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make CC="gcc" SCOTCHLIB=ptscotch						\
					libscotchmetis.a
make[4]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make[4]: `libscotchmetis.a' is up to date.
make[4]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make[3]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make[3]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make CC="gcc" SCOTCHLIB=ptscotch						\
					libscotchmetis.a
make[4]: Entering directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make[4]: `libscotchmetis.a' is up to date.
make[4]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
cp metis.h ../../include
cp libscotchmetis.a ../../lib
make[3]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src/libscotchmetis'
make[2]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh/scotch_5.1.12b/src'
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/create_serial_name_database.shared.o ../shared/create_serial_name_database.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/define_derivation_matrices.shared.o ../shared/define_derivation_matrices.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/detect_surface.shared.o ../shared/detect_surface.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/exit_mpi.shared.o ../shared/exit_mpi.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_attenuation_model.shared.o ../shared/get_attenuation_model.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_cmt.shared.o ../shared/get_cmt.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_element_face.shared.o ../shared/get_element_face.f90
make[1]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/decompose_mesh'
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_force.shared.o ../shared/get_force.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_jacobian_boundaries.shared.o ../shared/get_jacobian_boundaries.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_shape2D.shared.o ../shared/get_shape2D.f90
rm -f ../../lib/libmeshfem.a
ar cru ../../lib/libmeshfem.a ../../obj/mesh/safe_alloc_mod.shared.o ../../obj/mesh/store_coords.o ../../obj/mesh/read_mesh_parameter_file.o ../../obj/mesh/check_mesh_quality.o ../../obj/mesh/compute_parameters.o ../../obj/mesh/create_name_database.o ../../obj/mesh/create_regions_mesh.o ../../obj/mesh/create_visual_files.o ../../obj/mesh/define_subregions.o ../../obj/mesh/define_subregions_heuristic.o ../../obj/mesh/define_superbrick.o ../../obj/mesh/exit_mpi.o ../../obj/mesh/get_MPI_cutplanes_eta.o ../../obj/mesh/get_MPI_cutplanes_xi.o ../../obj/mesh/get_flags_boundaries.o ../../obj/mesh/get_global.o ../../obj/mesh/store_boundaries.o ../../obj/mesh/get_value_parameters.o ../../obj/mesh/hex_nodes.o ../../obj/mesh/meshfem3D.o ../../obj/mesh/param_reader.cc.o ../../obj/mesh/read_parameter_file.shared.o ../../obj/mesh/read_topo_bathy_file.o ../../obj/mesh/read_value_mesh_parameters.o ../../obj/mesh/read_value_parameters.shared.o ../../obj/mesh/save_databases.o ../../obj/mesh/utm_geo.o 
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_shape3D.shared.o ../shared/get_shape3D.f90
ranlib ../../lib/libmeshfem.a
mpiifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -o ../../bin/xmeshfem3D ../../obj/mesh/adios_manager_stubs.shared_noadios.o ../../obj/mesh/program_meshfem3D.o ../../obj/mesh/read_value_mesh_parameters.o ../../obj/mesh/get_value_parameters.o ../../lib/libmeshfem.a ../../obj/mesh/meshfem3D_adios_stubs.noadios.o ../../obj/mesh/parallel.o   ../../obj/mesh/safe_alloc_mod.shared.o
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/get_value_parameters.shared.o ../shared/get_value_parameters.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/gll_library.shared.o ../shared/gll_library.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/hex_nodes.shared.o ../shared/hex_nodes.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/lagrange_poly.shared.o ../shared/lagrange_poly.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/netlib_specfun_erf.shared.o ../shared/netlib_specfun_erf.f90
make[1]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/meshfem3D'
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/prepare_assemble_MPI.shared.o ../shared/prepare_assemble_MPI.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/read_topo_bathy_file.shared.o ../shared/read_topo_bathy_file.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/read_parameter_file.shared.o ../shared/read_parameter_file.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/read_value_parameters.shared.o ../shared/read_value_parameters.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/recompute_jacobian.shared.o ../shared/recompute_jacobian.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/save_header_file.shared.o ../shared/save_header_file.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/sort_array_coordinates.shared.o ../shared/sort_array_coordinates.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/utm_geo.shared.o ../shared/utm_geo.f90
ifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -c -o ../../obj/gen/write_VTK_data.shared.o ../shared/write_VTK_data.f90
make[1]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/specfem3D'
rm -f ../../lib/libgendatabases.a
ar cru ../../lib/libgendatabases.a ../../obj/gen/generate_databases_par.o ../../obj/gen/tomography_par.o ../../obj/gen/assemble_MPI_scalar.shared.o ../../obj/gen/calc_jacobian.o ../../obj/gen/fault_generate_databases.o ../../obj/gen/check_mesh_resolution.shared.o ../../obj/gen/create_name_database.shared.o ../../obj/gen/create_mass_matrices.o ../../obj/gen/create_regions_mesh.o ../../obj/gen/create_serial_name_database.shared.o ../../obj/gen/define_derivation_matrices.shared.o ../../obj/gen/detect_surface.shared.o ../../obj/gen/exit_mpi.shared.o ../../obj/gen/finalize_databases.o ../../obj/gen/generate_databases.o ../../obj/gen/get_absorbing_boundary.o ../../obj/gen/get_attenuation_model.shared.o ../../obj/gen/get_cmt.shared.o ../../obj/gen/get_coupling_surfaces.o ../../obj/gen/get_element_face.shared.o ../../obj/gen/get_force.shared.o ../../obj/gen/get_global.o ../../obj/gen/get_jacobian_boundaries.shared.o ../../obj/gen/get_model.o ../../obj/gen/get_MPI.o ../../obj/gen/get_perm_color.o ../../obj/gen/get_shape2D.shared.o ../../obj/gen/get_shape3D.shared.o ../../obj/gen/get_value_parameters.shared.o ../../obj/gen/gll_library.shared.o ../../obj/gen/hex_nodes.shared.o ../../obj/gen/lagrange_poly.shared.o ../../obj/gen/model_1d_cascadia.o ../../obj/gen/model_1d_prem.o ../../obj/gen/model_1d_socal.o ../../obj/gen/model_1d_layer.o ../../obj/gen/model_aniso.o ../../obj/gen/model_default.o ../../obj/gen/model_external_values.o ../../obj/gen/model_ipati.o ../../obj/gen/model_gll.o ../../obj/gen/model_salton_trough.o ../../obj/gen/model_tomography.o ../../obj/gen/netlib_specfun_erf.shared.o ../../obj/gen/param_reader.cc.o ../../obj/gen/pml_set_local_dampingcoeff.o ../../obj/gen/prepare_assemble_MPI.shared.o ../../obj/gen/read_topo_bathy_file.shared.o ../../obj/gen/read_parameter_file.shared.o ../../obj/gen/read_partition_files.o ../../obj/gen/read_value_parameters.shared.o ../../obj/gen/recompute_jacobian.shared.o ../../obj/gen/save_arrays_solver.o ../../obj/gen/save_header_file.shared.o ../../obj/gen/setup_color_perm.o ../../obj/gen/setup_mesh.o ../../obj/gen/sort_array_coordinates.shared.o ../../obj/gen/utm_geo.shared.o ../../obj/gen/write_VTK_data.shared.o ../../obj/gen/memory_eval.o 
ranlib ../../lib/libgendatabases.a
mpiifort  -O3 -DFORCE_VECTORIZATION -check nobounds -xHost -ftz -assume buffered_io -assume byterecl -align sequence -vec-report0 -std03 -diag-disable 6477 -implicitnone -warn truncated_source -warn argument_checking -warn unused -warn declarations -warn alignments -warn ignore_loc -warn usage -I../shared/  -o ../../bin/xgenerate_databases ../../obj/gen/program_generate_databases.o ../../obj/gen/parallel.o ../../obj/gen/adios_manager_stubs.shared_noadios.o ../../lib/libgendatabases.a ../../obj/gen/generate_databases_adios_stubs.noadios.o  
make[1]: Leaving directory `/global/home/kbai/GIT_repo/specfem3d_fault/src/generate_databases'
