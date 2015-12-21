#include <iostream>
extern "C"
{
	void compute_Diagonal_(realw*);
}
class conjugate_gradient
{
	conjugate_gradient(int NELE, realw* pdis, realw* pload, bool* Xsetfalse, bool* AXsetfalse); /** constructor*/
	~conjugate_gradient();





	private:
		int NDIM;
		int NSPEC;

		realw* displ; /**displ referred from displ in the solver*/
		realw* load; /**load referred from the load */
		realw* pdire; /**pdirection of conjugate gradient*/
		realw* residue; /**residue*/
		realw* Mprecon; /**preconditioner*/
		realw normold;
		bool* MASKX; /** for fixed displ boundary conditions*/
		bool* MASKAX; /** for fixed stress boundary conditions*/



}

conjugate_gradient::conjugate_gradient(int NELE, realw* pdis, realw* pload, bool* Xsetfalse, bool* AXsetfalse, bool precon )
{
	realw* W;
	NSPEC = NELE;
	/** reference to the outside array*/
	this->displ = pdis;
	this->load = pload;

	this->MASKX = Xsetfalse;
	this->MASKAX = AXsetfalse;

	pdire = new realw[3*NSPEC];


	if(precon)
		this.compute_precon();
}

void conjugate_gradient::compute_precon()
{
	this->Mprecon = new realw[3*NSPEC];
	compute_Diagonal_(&(this->Mprecon));   
}


	
