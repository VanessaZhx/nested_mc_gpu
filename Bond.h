#pragma once
#ifndef BOND_H
#define BOND_H

class Bond {

public:
	float bond_par;   // Par value of bond
	float bond_c;     // Coupon
	int bond_m;		  // Maturity
	float* bond_y;     // yeild curve
	float sigma;		// Sigma for generate random yield change

	Bond(float par, float c, int m, float* y, float sig) {
		this->bond_par = par;
		this->bond_c = c;
		this->bond_m = m;
		this->bond_y = y;
		this->sigma = sig;
	}
};

#endif // !BOND_H



