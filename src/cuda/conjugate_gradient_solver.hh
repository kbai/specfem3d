#include <iostream>
#include "mesh_constants_cuda.h"
#include <stdio.h>
extern "C"
{
	void compute_diagonal_(realw*);
}

extern "C"
{
	void compute_fault_gpu_(realw*, realw*, realw* ,realw*);
}

class conjugate_gradient
{
	public:

	conjugate_gradient(int NELE, realw* pdis, realw* pload, bool* Xsetfalse, bool* AXsetfalse, realw* gpu_displ, realw* gpu_force, bool precon, int proc_number); /** constructor*/
	
	~conjugate_gradient();

	void compute_precon();

	void checkfield(); /** for debug purpose*/

	void size();

	void gpu_init();

	void sum();

	void compute_forces();

	private:
		int NSPEC;

		realw* h_displ; /**displ referred from displ in the solver*/
		realw* h_load; /**load referred from the load */
		realw* h_residue; /**residue*/
		realw* h_Mprecon; /**preconditioner*/
		realw  h_normold;
		realw  h_sum;
		bool* h_MASKX; /** for fixed displ boundary conditions*/
		bool* h_MASKAX; /** for fixed stress boundary conditions*/

		realw* d_displ; /** gpu displ handle*/
		realw* d_force; /** gpu force handle*/
		realw* d_residue;
		
		realw* d_sum;   /** */
		realw* d_norm;  /** */
		realw* d_precon;
		realw* d_pdire; /**pdirection of conjugate gradient*/
		realw* d_MASKX;
		realw* d_MASKAX;
		int myrank;
};
