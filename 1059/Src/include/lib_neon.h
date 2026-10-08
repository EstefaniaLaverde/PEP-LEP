/** @file lib_neon.h
 * @brief Functions for calculations using neon instructions. If ARM architecture is detected by CMake before generation of the IDE project or Makefile, the files lib128.h, lib256.h, lib512.h and the corresponding .cpp files are not used for the generation of the project. 
 Instead the current file lib_neon.h and the corresponding .cpp file are used.
*/


#ifndef LIB_128_H
	#define LIB_128_H
#include "Data.h"
#include "ReadWrite.h"


/// <summary>
/// Constant global variable used for declaration of static arrays, containing different representation of the generator matrix. This parameter refers to the length of the code. 
/// The maximum possible length of the code depends on this value and the number of elements in the finite field.
/// </summary>
/// @note value should be multiple of 1536
static const int N_FIX = 3072; // N_FIX % (512*3) == 0

/// <summary>
/// Constant global variable used for declaration of static arrays, containing different representation of the generator matrix. This parameter referce to the dimension of the code.
/// The maximum possible dimension of the code depends on this value and the number of elements in the finite field.
/// </summary>
/// @note value should be multiple of 12
static const int K_FIX = 36; // K_FIX % (4*3) == 0


//constants for static matrices
// comments are the current values
static const int K_GF2 = K_FIX + 1; //36 + 1
static const int N_GF2 = N_FIX / 8; // 384
static const int N_CH2 = N_FIX; //  3072

static const int K_CH3 = K_FIX + 1; //36 + 1
static const int N_CH3 = N_FIX / 6; // 512

static const int K_P = K_FIX / 2 + 1; //18 + 1
static const int N_P = N_FIX * 2; // 6144


/// <summary>
/// Global variable for that contains weight spectrum
/// </summary>
extern unsigned long long int weights[];

//popcount functions and variables 
//static int POPCNT = 0;
//extern long long  popcount(unsigned long long  word);
//extern void popcnt_detect();


// global variables used in different functions
static int K = 0;
static int N = 0;
static int Q = 2;
static int M = 1; // for composite fields; gives the power of the characteristic of the field
static unsigned long long int w_searched = -1; // seves the searched weights
static bool less_than_flag = true; // flase if a word with weight less than the searched weight is found; the search stops
static bool equal_flag = true; // flase if a word with weight equal the searched weight is found; the search stops
static FILE* file; // stream to a file if the codewords need to be saved for future calculation

static int register_elements = ((N_FIX * 8 - 1) / 128) + 1; // gives the number of registers that will be used
static int Characteristic = 2;

static bool form = false; // if true -> write the codewords as a power of primitive element

// transition sequences of Q-ary Grey code for field with different characteristics
// used to show which copy of the generator matrix is used for calculation of next codeword

/// Transition sequences of Q-ary Grey code for field with characteristics 2
static short int TransitionSequence64[64] = { 0, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6,
						   1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1 };// used to show which copy of the generator matrix is used for calculation of next codeword
/// Transition sequences of Q-ary Grey code for field with characteristics 3
static short int TransitionSequence27[27] = { 0,1,1,2,1,1,2,1,1,3,1,1,2,1,1,2,1,1,3,1,1,2,1,1,2,1,1 };
/// Transition sequences of Q-ary Grey code for field with characteristics 5
static short int TransitionSequence25[25] = { 0,1,1,1,1, 2,1,1,1,1, 2,1,1,1,1, 2,1,1,1,1, 2,1,1,1,1 };
/// Transition sequences of Q-ary Grey code for field with characteristics 7
static short int TransitionSequence49[49] = { 0,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1 };



//main function defenitions

//-------------------weight calculation functions 128 ------------------------------//
/**
*@name WeightSpectrum
* @brief Calculation of the weight spectrum of a linear code using  neon instructions
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @note The weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
void calculateWeightCH3_neon(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m);

///Function for prime and composite fields using byte representation
void calculateWeightBytes_neon(dmat_type &generatorMatrix_byte, int n, int k, int m, int q);

///Function for fields with characteristic 2 using bitwise representation
void calculateWeightCH2_neon(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m);

///Function for GF2 using bitwise representation
void calculateWeightGF2_neon(dynamic_mat_short &generatorMatrix_bits, int n, int k);
/// @}


//------------------- Find word with w < w_fixed 128 ----------------------------------//
/**
*@name SearchLessThan
* @brief Searching for codeword with weight less than given value using  neon instructions
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
bool calculateWeightCH3_neon_less_than(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m, int w);

///Function for prime and composite fields using byte representation
bool calculateWeightBytes_neon_less_than(dmat_type &generatorMatrix_byte, int n, int k, int m, int q,  int w);

///Function for fields with characteristic 2 using bitwise representation
bool calculateWeightCH2_neon_less_than(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m, int w);

///Function for GF2 using bitwise representation
bool calculateWeightGF2_neon_less_than(dynamic_mat_short &generatorMatrix_bits, int n, int k, int w);
/// @}

//------------------- Find word with w == w_fixed 128 ----------------------------------//

/**
*@name SearchEqualTo
* @brief Searching for codeword with weight equal to given value using  neon instructions
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
bool calculateWeightCH3_neon_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int d);

///Function for prime and composite fields using byte representation
bool calculateWeightBytes_neon_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int d);

///Function for fields with characteristic 2 using bitwise representation
bool calculateWeightCH2_neon_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int d);

///Function for GF2 using bitwise representation
bool calculateWeightGF2_neon_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int d);
/// @}

//------------------- Count and write word with w == w_fixed 128 ----------------------------------//
/**
*@name CountEqualTo
* @brief Counting the number of codewords with weight equal to given value using  neon instructions and writing the generated nonproportional codewords in file **Result_codewords.txt**
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
unsigned long long int  calculateNumberOfWordsCH3_neon_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm);

///Function for prime and composite fields using byte representation
unsigned long long int  calculateNumberOfWordsBytes_neon_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm);

///Function for fields with characteristic 2 using bitwise representation
unsigned long long int calculateNumberOfWordsCH2_neon_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int w, bool multiplicativeForm);

///Function for GF2 using bitwise representation
unsigned long long int calculateNumberOfWordsGF2_neon_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k,  int w, bool multiplicativeForm);
/// @}

//------------------- Count and write word with w < w_fixed 128 ----------------------------------//

/**
*@name CountLessThan
* @brief Counting the number of codewords with weight less than given value using  neon instructions registers and writing the generated nonproportional codewords in file **Result_codewords.txt**
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
unsigned long long int calculateNumberOfWordsCH3_neon_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int w, bool multiplicativeForm);

///Function for prime and composite fields using byte representation
unsigned long long int  calculateNumberOfWordsBytes_neon_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q,  int w, bool multiplicativeForm);

///Function for fields with characteristic 2 using bitwise representation
unsigned long long int calculateNumberOfWordsCH2_neon_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int w, bool multiplicativeForm);

///Function for GF2 using bitwise representation
unsigned long long int calculateNumberOfWordsGF2_neon_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k,  int w, bool multiplicativeForm);
/// @}
#endif
