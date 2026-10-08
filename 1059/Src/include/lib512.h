/** @file lib512.h
 * @brief Functions for calculations using 512-bit registers
*/

#ifndef LIB_512_H
#define LIB_512_H
#include "Data.h"
#include "lib256.h"
#include "lib128.h"
#include "ReadWrite.h"
#include "Polynomials.h"

//-------------------weight calculation functions 512 ------------------------------//

/**
*@name WeightSpectrum
* @brief Calculation of the weight spectrum of a linear code using 512-bit registers
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @note The weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
void calculateWeightCH3_512(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m);

///Function for prime and composite fields using byte representation
void calculateWeightBytes_512(dmat_type& generatorMatrix_byte, int n, int k, int m, int q);

///Function for fields with characteristic 2 using bitwise representation
void calculateWeightCH2_512(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m);

///Function for GF2 using bitwise representation
void calculateWeightGF2_512(dynamic_mat_short& generatorMatrix_bits, int n, int k);
///@}


//------------------- Find word with w < w_fixed 512 ----------------------------------//

/**
*@name SearchLessThan
* @brief Searching for codeword with weight less than given value using 512-bit registers
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @param w the value of the searched weight
* @retval true if a codeword with weight **w** is found
* @note Partial weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
bool calculateWeightCH3_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w);

///Function for prime and composite fields using byte representation
bool calculateWeightBytes_512_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w);

///Function for fields with characteristic 2 using bitwise representation
bool calculateWeightCH2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w);

///Function for GF2 using bitwise representation
bool calculateWeightGF2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w);

//------------------- Find word with w == w_fixed 512 ----------------------------------//
/**
*@name SearchEqualTo
* @brief Searching for codeword with weight equal to given value using 256-bit registers
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @param w the value of the searched weight
* @retval true if a codeword with weight **w** is found
* @note Partial weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
bool calculateWeightCH3_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int d);

///Function for prime and composite fields using byte representation
bool calculateWeightBytes_512_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int d);

///Function for fields with characteristic 2 using bitwise representation
bool calculateWeightCH2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int d);

///Function for GF2 using bitwise representation
bool calculateWeightGF2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int d);

//------------------- Count and write word with w == w_fixed 512 ----------------------------------//
/**
*@name CountEqualTo
* @brief Counting the number of codewords with weight equal to given value using 256-bit registers and writing the generated nonproportional codewords in file **Result_codewords.txt**
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @param w the value of the searched weight
* @param multiplicativeForm if true, the codewords will be written, using multiplicative representation of the elements of the field
* @retval Total number of codewords with weight **w**
* @note  Weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
unsigned long long int calculateNumberOfWordsCH3_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm);

///Function for prime and composite fields using byte representation
unsigned long long int calculateNumberOfWordsBytes_512_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm);

///Function for fields with characteristic 2 using bitwise representation
unsigned long long int calculateNumberOfWordsCH2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm);

///Function for GF2 using bitwise representation
unsigned long long int calculateNumberOfWordsGF2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w, bool multiplicativeForm);

//------------------- Count and write word with w < w_fixed 512 ----------------------------------//
/**
*@name CountLessThan
* @brief Counting the number of codewords with weight less than given value using 256-bit registers and writing the generated nonproportional codewords in file **Result_codewords.txt**
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @param w the value of the searched weight
* @param multiplicativeForm if true, the codewords will be written, using multiplicative representation of the elements of the field
* @retval Total number of codewords with weight less than **w**
* @note  Weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
unsigned long long int calculateNumberOfWordsCH3_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm);

///Function for prime and composite fields using byte representation
unsigned long long int calculateNumberOfWordsBytes_512_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm);

///Function for fields with characteristic 2 using bitwise representation
unsigned long long int calculateNumberOfWordsCH2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm);

///Function for GF2 using bitwise representation
unsigned long long int calculateNumberOfWordsGF2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w, bool multiplicativeForm);

#endif
