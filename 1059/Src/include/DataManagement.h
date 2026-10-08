
#ifndef DATA_MANAGEMENT_H
	#define DATA_MANAGEMENT_H
#include "Data.h"

// using bitwise representation 
//fields with characteristic 2
void num_to_coef_gf2( int& k, int& n, int m);
void num_to_coef_gf4(int& k, int& n, int m);
void num_to_coef_gf8(int& k, int& n, int m);
void num_to_coef_gf16(int& k, int& n, int m);
void num_to_coef_gf32(int& k, int& n, int m);
void num_to_coef_gf64(int& k, int& n, int m);

//fields with characteristic 3
void num_to_coef_gf27(int k, int n, int m);
void num_to_coef_gf9(int k, int n, int m);
void num_to_coef_gf3(int k, int n, int m);

// using byte representation
void num_to_coef_char2(int& k, int& n, int m);
void num_to_coef_char25(int& k, int& n, int m);
void num_to_coef_char49(int& k, int& n, int m);

#endif 

