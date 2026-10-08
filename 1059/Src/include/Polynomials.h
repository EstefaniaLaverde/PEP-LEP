/** @file Polinomials.h
 * @brief Declaration of structure and functions used in computations in composite fields.
 */
#ifndef POLINOMIALS_H
#define POLINOMIALS_H
#include "DataManagement.h"
//#include "Data.h"


/** @struct polinom
 *  @brief Polinomial structure used for composite finite fields
 *  @var polinom::coef
 *  Coeficients of the polinomial
 *  @var polinom::q
 *  number of elements in the field
 * @var polinom::grad
 * degree of a polynomial
 */

struct polinom {
	int coef[8],///coeficients of the polinomial
		q, /// number of elements in the field
		grad; /// degree of a polynomial
};

/// <summary>
/// Structure representing elements of composite finite field
/// </summary>
struct comp_elements {
	/// element of the field using polinomial representation
	polinom polynomial; 
	/// element of the field using decimal representation
	int decimal;
	/// element of the field using multiplycative representation (stored as a power of a primitive element)
	int power_of_prim; 
};

int multipl_to_dec(int power);
int dec_to_multipl(int decimal);

int polynomial_to_multipl(polinom& p);
int politoint(polinom& a, int q);


/// <summary>
/// functions for generating multiplication tables for a given finite field with q elements
/// </summary>
/// <param name="q">number of elements in the finite field</param>
void maketable(int q);
int multiply(int m, int n);

//functions for multiplying the generator matrix with q used for composite field
void multiplyGF9(int k, int n);
void multiplyGF4(int k, int n);
void multiplyGF8(int k, int n);
void multiplyGF16( int k, int n);
void multiplyGF25(int k, int n);
void multiplyGF49( int k, int n);
void multiplyGF32(int k, int n);
void multiplyGF64(int k, int n);
void multiplyGF27(int k, int n);

//function for addition of two vectors of the generator matrix
//used in calculation without registers
int add_vector(int n, int row_H, int row, int q1);


extern int CHI;
#endif // !POLINOMIALS_H


