#include "mesh_constants_cuda.h"
#include <stdio.h>
extern "C"
{
	void compute_diagonal_(realw*);
}

extern "C"
{
	void compute_fault_gpu_(realw*, realw*, int* ,int*);
}

extern "C"
{
	void compute_fault_gpu_local_(realw*, realw*, int* , int*);
}

extern "C"
{
	void sum_all_all_cr_(realw* sum, realw* sum_all);
}

class conjugate_gradient
{
	public:

	conjugate_gradient(int NELE, realw* pdis, realw* pload, int* Xsetfalse, int* AXsetfalse, realw* gpu_displ, realw* gpu_force, int precon, int proc_number, realw* recorder, int* MPI_repeat, realw* h_restri_op, realw* rmass); /** constructor*/
	
	~conjugate_gradient();

	void reinit(realw deltat_sq_inv_in);

	void compute_precon();

	void checkfield(); /** for debug purpose*/

	void size();

	void gpu_init();

	void sum();

	void compute_forces(realw* disp_field);

	void compute_forces_local(realw* disp_field);

	void update_val_dire();

	void compute_forces_call();
	
	void compute_precond(realw* disp_field, realw* re_field);
	
	void get_field_from_gpu();

	void solve(realw tolerance);

	realw* d_load;
	
	int* d_MASKX;

	int* d_MASKAX;

	private:
		int NSPEC;

		realw* h_displ; /**displ referred from displ in the solver*/
		realw* h_load; /**load referred from the load */
		realw* h_rec;
		realw* h_residue; /**residue*/
		realw* h_blocksum;
		realw* h_Mprecon; /**preconditioner*/
		realw  h_normold;
		realw  h_sum;
		realw  h_sum_all;
		realw  deltat_sq_inv;
		int* h_MASKX; /** for fixed displ boundary conditions*/
		int* h_MASKAX; /** for fixed stress boundary conditions*/
		int* d_MPI_repeat; /** for storing bool of whether the element is repeated*/

		realw* d_displ; /** gpu displ handle*/
		realw* d_force; /** gpu force handle*/
//		realw* d_load;
		realw* d_residue;
		
		realw* d_sum;   /** */
		realw* d_blocksum; 
		realw* d_norm;  /** */
		realw* d_precon;
		realw* d_pdire; /**pdirection of conjugate gradient*/
//		realw* d_MASKX;
//		realw* d_MASKAX;
		realw* d_tmp; /** temp gpu vector for storing intermediate results */
		realw* d_tmp2;
		realw* d_restri_op; /** restriction operator for schwartz method*/
		realw* rmass;	/** these are all references to mesh struct object */
		realw  r0; /** initial residue, used to terminate the iteration if certain criterion is meet*/
		int myrank;

};

__device__ realw dsum;
