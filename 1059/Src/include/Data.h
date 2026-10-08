/** @file Data.h
@brief Declaration of global variables and structures used in the calculations. 
*/
#ifndef DATA_H
#define DATA_H
#include <cstdlib>
#include <cstdio>

//defining dynamic structures
//for the generator matrix


typedef unsigned long long int* dynamic_row_short; 
dynamic_row_short drow_short_new(int size); // function for alloczating dynamic row for the generator matrix (unsigned long long int)
void drow_short_free(dynamic_row_short r); // function for freeing dynamic row

/// <summary>
/// Definition of dynamic structure describing generator matrix of a linear code saved as dynamic array of unsigned long long int elements
/// </summary>
typedef struct  _dmatlu_type dynamic_mat_short;

struct _dmatlu_type { // structure for dynamic generator matrix (unsigned long long int)
	int n, k, nreal, kreal, q, mem = 0, num = 0;
	dynamic_row_short* a;
};

dynamic_mat_short dmat_short_newh(dynamic_mat_short& c, int k, int n, int q);
void dmat_short_free(dynamic_mat_short& c);
dynamic_mat_short dmat_short_new(dynamic_mat_short& c, int k, int n, int q);


typedef unsigned char* drow;

drow drow_new(int size); // function for allocating dynamic row for generator matrix (char)
void drow_free(drow& r); // function for freeing  dynamic row for generator matrix (char)

/// <summary>
/// Definition of dynamic structure describing generator matrix of a linear code saved as dynamic array of char elements
/// </summary>
typedef struct _dmat_type dmat_type;

struct _dmat_type { // structure for dynamic generator matrix (char)
	int n=0, k=0, nreal=0, kreal=0, q=0, mem = 0;
	char name[50]={};
	drow* a=NULL;      
};

dmat_type dmat_newh(dmat_type& c, int m, int n, int q);
dmat_type dmat_new(dmat_type& c, int m, int n, int q);
void dmat_free(dmat_type& c);

/// Dynamic matrix for calculation when using byte representation 
extern dmat_type bitsCharCF; 

/// Dynamic matrix for calculation when using bit representation 
extern dynamic_mat_short bits; 

/// Dynamic matrix for used to save the read matrix 
extern dynamic_mat_short matrix;

/// Dynamic matrix for used for calculation without registers 
extern dynamic_mat_short matrixH; 

/// <summary>
/// Global variable that is used to force the use of a specific instruction set for testing purposes. 
/// </summary>
/// @note if value > 0, then the value of instructionSet parameter, defined in LinCodeWeightInv.h, is set to the same value
extern int test; 

//--------------------- Data Management functions----------------------//
// using bitwise representation 
//fields with characteristic 2
void num_to_coef_gf2(int& k, int& n, int m);
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

#ifndef ERRORQ
#define ERRORQ(expr) \
        if ((expr)) { \
		FILE *fran;\
	    fran=fopen("error.txt", "a");\
		fprintf(fran,"ERROR %s: line %d: assertion failed: " \
			"(%s)\n",__FILE__,__LINE__,#expr); \
			int i=fclose(fran); \
			exit(EXIT_FAILURE);\
			int t;\
           t = scanf("%d", &t);\
	while(true)\
	{\
		printf("ERROR");\
        	        }\
			exit(EXIT_FAILURE);\
	}
#endif /* !ASSERT */


#endif // !DATA_H
