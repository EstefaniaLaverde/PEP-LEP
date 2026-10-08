/** @file lib128.h
 * @brief Functions for calculations using 128-bit registers. The defined global variables and functions are also used in other places in the library as external components.
*/

#ifndef LIB_128_H
	#define LIB_128_H
#include "Data.h"
#include "ReadWrite.h"
#include "Polynomials.h"

/// <summary>
/// Global variable for that contains weight spectrum
/// </summary>
extern unsigned long long int weights[];

/// <summary>
/// Global variable set by popcnt_detect() to show whether the current architecture has a **popcnt** instruction
/// </summary>
/// @note If value < 0 then the architecture does not have a **popcnt** instruction. If value = 1 then the architecture has a **popcnt** instruction for a computer word.
/// If value = 2 then the architecture has a 512-bit **popcnt** instructions. This parameter is used in computations with 128, 256 and 512-bit registers.
extern int POPCNT;

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
/// Function that detects whether the current architecture has a **popcnt** instruction
/// </summary>
/// @note This function is used in computations with 128, 256 and 512-bit registers. The implementation is in the corresponding .cpp file.
extern void popcnt_detect();

/// <summary>
/// Function that based on the used compiler computes the number of nonzero bits in a computer word. If POPCNT<0 it uses masks for the calculations.
/// </summary>
/// <param name="word">A 64-bit computer word of type unsigned long long </param>
/// <returns>The number of nonzero bits in **word**</returns>
/// @note This function is used in computations with 128, 256 and 512-bit registers. The implementation is in the corresponding .cpp file.
extern long long  popcount(unsigned long long  word);



// global variables used in different functions
static int K = 0;
static int N = 0;
static int Q = 2;
static int M = 1; // for composite fields; gives the power of the characteristic of the field
static int w_searched = -1; // seves the searched weights
static bool less_than_flag = true; // flase if a word with weight less than the searched weight is found; the search stops
static bool equal_flag = true; // flase if a word with weight equal the searched weight is found; the search stops
static FILE* file; // stream to a file if the codewords need to be saved for future calculation

static int register_elements = ((N_FIX * 8 - 1) / 128) + 1; // gives the number of registers that will be used
static int Characteristic = 2;

static bool form = false; // if true -> write the codewords as a power of primitive element

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
* @brief Calculation of the weight spectrum of a linear code using 128-bit registers
* @param generatorMatrix_bits Generator matrix with appropriate representation of the elements of the field, saved in a dynamic structure, defined in Data.h
* @param n length of the code
* @param k dimension of the code
* @param p characteristic of the field; not present for the functions for field with set characteristic
* @param m calculations for finite field with p^m elements
* @note The weight spectrum is saved in a global array **weights**
*/
///@{

///Function for fields with characteristic 3
void calculateWeightCH3_128(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m);

///Function for prime and composite fields using byte representation
void calculateWeightBytes_128(dmat_type &generatorMatrix_byte, int n, int k, int m, int p);

///Function for fields with characteristic 2 using bitwise representation
void calculateWeightCH2_128(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m);

///Function for GF2 using bitwise representation
void calculateWeightGF2_128(dynamic_mat_short &generatorMatrix_bits, int n, int k);
/// @}


//------------------- Find word with w < w_fixed 128 ----------------------------------//
/**
*@name SearchLessThan
* @brief Searching for codeword with weight less than given value using 128-bit registers
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
bool calculateWeightCH3_128_less_than(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m, int w);

///Function for prime and composite fields using byte representation
bool calculateWeightBytes_128_less_than(dmat_type &generatorMatrix_byte, int n, int k, int m, int p,  int w);

///Function for fields with characteristic 2 using bitwise representation
bool calculateWeightCH2_128_less_than(dynamic_mat_short &generatorMatrix_bits, int n, int k, int m, int w);

///Function for GF2 using bitwise representation
bool calculateWeightGF2_128_less_than(dynamic_mat_short &generatorMatrix_bits, int n, int k, int w);
/// @}

//------------------- Find word with w == w_fixed 128 ----------------------------------//

/**
*@name SearchEqualTo
* @brief Searching for codeword with weight equal to given value using 128-bit registers
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
bool calculateWeightCH3_128_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int d);

///Function for prime and composite fields using byte representation
bool calculateWeightBytes_128_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int p, int d);

///Function for fields with characteristic 2 using bitwise representation
bool calculateWeightCH2_128_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int d);

///Function for GF2 using bitwise representation
bool calculateWeightGF2_128_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int d);
/// @}


//------------------- Count and write word with w == w_fixed 128 ----------------------------------//


/**
*@name CountEqualTo
* @brief Counting the number of codewords with weight equal to given value using 128-bit registers and writing the generated nonproportional codewords in file **Result_codewords.txt**
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
unsigned long long int  calculateNumberOfWordsCH3_128_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm);

///Function for prime and composite fields using byte representation
unsigned long long int  calculateNumberOfWordsBytes_128_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int p, int w, bool multiplicativeForm);

///Function for fields with characteristic 2 using bitwise representation
unsigned long long int calculateNumberOfWordsCH2_128_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int w, bool multiplicativeForm);

///Function for GF2 using bitwise representation
unsigned long long int calculateNumberOfWordsGF2_128_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k,  int w, bool multiplicativeForm);
/// @}


//------------------- Count and write word with w < w_fixed 128 ----------------------------------//


/**
*@name CountLessThan
* @brief Counting the number of codewords with weight less than given value using 128-bit registers and writing the generated nonproportional codewords in file **Result_codewords.txt**
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
unsigned long long int calculateNumberOfWordsCH3_128_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int w, bool multiplicativeForm);

///Function for prime and composite fields using byte representation
unsigned long long int  calculateNumberOfWordsBytes_128_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int p,  int w, bool multiplicativeForm);

///Function for fields with characteristic 2 using bitwise representation
unsigned long long int calculateNumberOfWordsCH2_128_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m,  int w, bool multiplicativeForm);

///Function for GF2 using bitwise representation
unsigned long long int calculateNumberOfWordsGF2_128_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k,  int w, bool multiplicativeForm);
/// @}

#endif
