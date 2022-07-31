#pragma once
#ifndef BARRIER_OPTION
#define BARRIER_OPTION

#include "Stock.h"

class BarrierOption {
public:
	// underlying stock
	Stock* s;

	// for barrier
	float k;	// Execution price
	float h;	// Barrier

	BarrierOption(Stock *s, float k, float h) {
		this->s = s;
		this->k = k;
		this->h = h;
	}
};

#endif
