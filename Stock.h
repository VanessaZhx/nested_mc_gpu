#pragma once

class Stock{

public:
	float s0;
	float mu;
	float var;
	int x;


	Stock(float s0, float mu, float var, int x) {
		this->s0 = s0;
		this->mu = mu;
		this->var = var;
		this->x = x;
	}

	
};

