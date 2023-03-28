#include <stdio.h>
#include <iostream>
#include "geometry.h"
/*
#include <fsstream.h>
*/

using namespace std;

// Random number in [0,1)
#define RAN ((double )random() * 4.6566128752457969e-10)
const double PI = 3.1415926536;


const int NCHAIN = 20;            // Number of chains
const int N = 128;                // Chain length
const int TYPE_MONO = 3;        // Monomer type
const int TYPE_BOND = 1;        // Bond type
const int TYPE_ANGLE = 1;       // Angle types
const double LBOND = 0.97;      // Bond length




typedef struct {
    Coord r;    // Monomer coordinate
    int ix;     // Box indices ..
    int iy;     // for periodic ..
    int iz;     // boundaries
    int atomType;
    int bondType;
    int id;
} Monomer_tp;   // Monomer struct


typedef struct {
    int nMono;
    Monomer_tp mon[N];
} Chain_tp;    // Chain struct


extern Coord ranVec();
extern void periodicBoundary(double d, Chain_tp &chain);
extern void periodicXYBoundary(double d, Chain_tp &chain);
extern int modulo(double a, double b);
   


int main() {

    Chain_tp chain[NCHAIN];    // Array of chains 


    Coord dr, v1, v2;
    FILE *fp_lammps, *fp_fixed;
    fp_lammps = fopen("system.xyz","w");

    double x, y, z;
    int nAtoms;


    nAtoms = NCHAIN*N;

    int nBonds = NCHAIN*(N-1);
    int nAngle = NCHAIN*(N-2);

    
    double Lbox = 200;

    // Setup box corners:
    Coord lowCorner(-Lbox/2, -Lbox/2, -Lbox/2);
    Coord hiCorner(Lbox/2, Lbox/2, Lbox/2);


    



    int c_chain = 0;
    int c_atom = 1;    
    for (int i = 0; i < NCHAIN; i++) {
      int c_mon = 0;
      int type_atom = 1;            // anchor-atom, type 1
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      chain[c_chain].nMono = N;
      chain[c_chain].mon[c_mon].r[0] = x;
      chain[c_chain].mon[c_mon].r[1] = y;
      chain[c_chain].mon[c_mon].r[2] = z;
      chain[c_chain].mon[c_mon].atomType = type_atom;
      chain[c_chain].mon[c_mon].bondType = 1;
      chain[c_chain].mon[c_mon].id = c_atom;
      c_mon++;
      c_atom++;
      for (int n = 1; n < N; n++) {
	type_atom = 2;                  // normal atoms have type 2
	if (n == 1) type_atom = 1;      // fix another one!
	if (n == N-1) type_atom = 3;    // end-atom has type 3
	dr = 0.97*ranVec();
	chain[c_chain].mon[c_mon].r = chain[c_chain].mon[c_mon-1].r + dr;
	chain[c_chain].mon[c_mon].atomType = type_atom;
	chain[c_chain].mon[c_mon].bondType = 1;
	chain[c_chain].mon[c_mon].id = c_atom;
	c_mon++;
	c_atom++;
      }
      c_chain++;
    }

   //Introduce periodic boundary conditions:
    for (int k = 0; k < NCHAIN; k++) {
        periodicBoundary(Lbox, chain[k]);
    }


    // Write out LAMMPS file:
    fprintf(fp_lammps, "LAMMPS FENE chain data file\n");
    fprintf(fp_lammps, "\n");
    fprintf(fp_lammps, "%d      atoms\n", nAtoms);
    fprintf(fp_lammps, "%d      bonds\n", nBonds);
    fprintf(fp_lammps, "%d      extra bond per atom\n", 2); 
    fprintf(fp_lammps, "%d      angles\n", nAngle);
    fprintf(fp_lammps, "%d      dihedrals\n", 0);
    fprintf(fp_lammps, "%d      impropers\n", 0);
    fprintf(fp_lammps, "\n");
    fprintf(fp_lammps, "%d      atom types\n", TYPE_MONO);
    fprintf(fp_lammps, "%d      bond types\n", TYPE_BOND);
    fprintf(fp_lammps, "%d      angle types\n", TYPE_ANGLE);
    fprintf(fp_lammps, "%d      dihedral types\n", 0);
    fprintf(fp_lammps, "%d      improper types\n", 0);
    fprintf(fp_lammps, "\n");
    fprintf(fp_lammps, "%lf %lf xlo xhi\n", lowCorner[0], hiCorner[0]);
    fprintf(fp_lammps, "%lf %lf ylo yhi\n", lowCorner[1], hiCorner[1]);
    fprintf(fp_lammps, "%lf %lf zlo zhi\n", lowCorner[2], hiCorner[2]);
    fprintf(fp_lammps, "\n");    
    fprintf(fp_lammps, "Masses\n");
    fprintf(fp_lammps, "\n");
    for (int i = 0; i < TYPE_MONO; i++) {
	fprintf(fp_lammps, "%d      %lf\n", i+1, 1.0);
    }
    fprintf(fp_lammps, "\n");    
    fprintf(fp_lammps, "Atoms\n");
    fprintf(fp_lammps, "\n");


    c_chain = 0;
    c_atom = 0;
    for (int i = 0; i < NCHAIN; i++) {
      int c_mon = 0;
      for (int n = 0; n < N; n++) {
	fprintf(fp_lammps, "%d  %d  %d  %lf  %lf  %lf  %d  %d  %d\n",
		chain[c_chain].mon[c_mon].id, c_chain+1, 
		chain[c_chain].mon[c_mon].atomType, 
		chain[c_chain].mon[c_mon].r[0], 
		chain[c_chain].mon[c_mon].r[1], 
		chain[c_chain].mon[c_mon].r[2],
		chain[c_chain].mon[c_mon].ix,
		chain[c_chain].mon[c_mon].iy,
		chain[c_chain].mon[c_mon].iz);
	c_atom++;
	c_mon++;
      }
      c_chain++;
    }


    fprintf(fp_lammps, "\n");    
    fprintf(fp_lammps, "Bonds\n");
    fprintf(fp_lammps, "\n");
    int c_bond = 1;
    c_chain = 0;
    for (int i = 0; i < NCHAIN; i++) {
      int c_mon = 1;
      for (int n = 1; n < N; n++) {
	fprintf(fp_lammps, "%d  %d  %d  %d\n",
		c_bond, 1, 
		chain[c_chain].mon[c_mon-1].id,
		chain[c_chain].mon[c_mon].id);
	c_bond++;
	c_mon++;
      }
      c_chain++;          
    }

    cout << "nBond: " << c_bond-1 << endl;
    cout << "Should be: " << nBonds << endl;

    fprintf(fp_lammps, "\n");
    fprintf(fp_lammps, "Angles\n");
    fprintf(fp_lammps, "\n");
    int c_angle = 1;
    c_chain = 0;
    for (int i = 0; i < NCHAIN; i++) {
      int c_mon = 2;
      for (int n = 2; n < N; n++) {
	fprintf(fp_lammps, "%d  %d  %d  %d  %d\n",
                c_angle, 1,
		chain[c_chain].mon[c_mon-2].id,
                chain[c_chain].mon[c_mon-1].id,
                chain[c_chain].mon[c_mon].id);
	c_angle++;
        c_mon++;
      }
      c_chain++;
    }




    fclose(fp_lammps);

    cout << "That's all folks! " << endl;
    exit(0);

}


Coord ranVec() {
    // Create a random vector of unit-length
    double x, y, z, norm2;
    do {
	x = 2.0*RAN - 1.0;
	y = 2.0*RAN - 1.0;
	z = 2.0*RAN - 1.0;
        norm2 = x*x + y*y + z*z;
    } while (norm2 > 1.0 || norm2 < 1.0e-12);
    Coord v(x, y, z);
    v /= sqrt(norm2);
    return v;
}

void periodicBoundary(double d, Chain_tp &chain) {
    // Apply periodic boundary conditions 
    int image_x, image_y, image_z;
    
    // The first monomer is always inside the box:
    chain.mon[0].ix = 0;
    chain.mon[0].iy = 0;
    chain.mon[0].iz = 0;

    // Check other monomers:
    for (int n = 1; n < chain.nMono; n++) {
	image_x = modulo(chain.mon[n].r[0], d);
	image_y = modulo(chain.mon[n].r[1], d);
	image_z = modulo(chain.mon[n].r[2], d);
        chain.mon[n].r[0] -= image_x*d;  // re-map monomer
        chain.mon[n].r[1] -= image_y*d;
        chain.mon[n].r[2] -= image_z*d;
        chain.mon[n].ix = image_x;       // remember image 
        chain.mon[n].iy = image_y;
        chain.mon[n].iz = image_z;
    }
}

void periodicXYBoundary(double d, Chain_tp &chain) {
    // Apply periodic boundary conditions in X and Y 
    int image_x, image_y, image_z;
    
    // The first monomer is always inside the box:
    chain.mon[0].ix = 0;
    chain.mon[0].iy = 0;
    chain.mon[0].iz = 0;

    // Check other monomers:
    for (int n = 1; n < chain.nMono; n++) {
	image_x = modulo(chain.mon[n].r[0], d);
	image_y = modulo(chain.mon[n].r[1], d);
	//image_z = modulo(chain.mon[n].r[2], d);
        image_z = 0;                     // no image box in z 
        chain.mon[n].r[0] -= image_x*d;  // re-map monomer
        chain.mon[n].r[1] -= image_y*d;
        chain.mon[n].r[2] -= image_z*d;
        chain.mon[n].ix = image_x;       // remember image 
        chain.mon[n].iy = image_y;
        chain.mon[n].iz = image_z;
    }
}

int modulo(double a, double b) {
    if (fabs(a) < b/2.0) {
	return 0;   // particle is inside box
    }
    else {
	if (a >= 0.0) {
	    a -= b/2.0;
            return (int )(a/b) + 1;
	}
        if (a < 0.0){
	    a += b/2.0;  
	    return (int )(a/b) - 1;
	}
    }
}







