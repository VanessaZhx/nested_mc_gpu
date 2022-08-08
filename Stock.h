#pragma once

class Stock{

public:
	float s0;
	float mean;
	float std;
	int x;


	Stock(float s0, float mean, float std, int x) {
		this->s0 = s0;
		this->mean = mean;
		this->std = std;
		this->x = x;
	}

	
};

