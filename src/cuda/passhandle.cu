#include <stdio.h>
#include <config.h>
#include <stdlib.h>
#include <mpi.h>
#include "mesh_constants_cuda.h"
#include "conjugate_gradient_solver.hh"
extern "C"
void FC_FUNC_(passhandle,
		      PASSHANDLE)(long* Mesh_pointer,long* CG_pointer, int* NELE, realw* disp, realw* pload, bool*Xsetfalse, bool* AXsetfalse, int* myrank  )
{
	Mesh* mp = (Mesh*)(*Mesh_pointer);
	printf("\nMYRANK:%d\n",*myrank);
	conjugate_gradient *newCG = new conjugate_gradient(*NELE , disp, pload, Xsetfalse, AXsetfalse, mp->d_displ, mp->d_accel, true, *myrank);
	newCG->checkfield();
	MPI_Barrier(MPI_COMM_WORLD);
	newCG->sum();
	*CG_pointer = (long)(newCG); 
	newCG->compute_forces();
//	abort();
}	
