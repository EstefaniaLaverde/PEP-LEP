/** @file LinCodeWeightInv.h
 * @brief Declaration of end user functions.
 */

#ifndef WEIGHT_DISTRIBUTION_LIB_SSE_H
#define WEIGHT_DISTRIBUTION_LIB_SSE_H
//#include <iostream>


/// <summary>
/// Global array, containing the (partial) weight distribution of a code, after calculations.
/// </summary>
extern unsigned long long int weights[];

/// <summary>
/// Global variable that shows the instruction set that is used.
/// </summary>
/// @note if value >= 9 then the library uses AVX512 instruction set
/// @note if value >= 8 then the library uses AVX2 instruction set
/// @note if value >= 5 then the library uses SSE4.1 instruction set
/// @note if value < 5 then computations are executed without extended registers 
extern int instructionSet;

/// <summary>
/// Global function that detects the available instruction set on the specific hardware. It sets the value of **instructionSet** parameter. 
/// </summary>
/// @note ARM architecture is detected by using predefined macros.
extern void detect();

/// Calculation of the weight spectrum of a linear code for a given generator matrix
/// @param generatorMatrix Generator matrix, saved in 2-dimensional array
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param multiplicativeForm value=true if the elements of the field are written in multiplicative form, value=false otherwise
/// @note The weight distribution is written in the global variable weights
void calculateWeightDistribution(int** generatorMatrix, int N, int K, int Q, bool multiplicativeForm);

/// Calculation of the weight spectrum of a randomly generated linear code for given parameters of the code
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @note The weight distribution is written in the global variable weights
void calculateWeightDistribution(int N, int K, int Q);

/// Calculating the weight distribution of linear codes for given generator matrices, written in a file
/// @param generatorMatrixFile Name of the file, containing the generator matrices
/// @note The weight distribution is written in output file "Weight_dis.txt"
void calculateWeightDistribution(char* generatorMatrixFile);

/// Calculating the minimum distance of a linear code for a given generator matrix
/// @param generatorMatrix Generator matrix, saved in 2-dimensional array
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param multiplicativeForm value=true if the elements of the field are written in multiplicative form, value=false otherwise
/// @returns minimum distance of the code as unsigned long long int value
unsigned long long int min_dis(int** generatorMatrix, int N, int K, int Q, bool multiplicativeForm);

/// Calculating the minimum distance of a randomly generated linear code for given parameters of the code
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @returns minimum distance of the code as unsigned long long int value
unsigned long long int min_dis(int N, int K, int Q);

/// Calculating the minimum distance of linear codes for given generator matrices, written in a file
/// @param generatorMatrixFile Name of the file, containing the generator matrices
/// @note minimum distance of the linear codes are written in a output file "min_distance.txt"
void min_dis(char* generatorMatrixFile);

/// Searching  for a codeword with weight less than a given value of a linear code for a given generator matrix
/// @param generatorMatrix Generator matrix, saved in 2-dimensional array
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
/// @param multiplicativeForm true if the elements of the field are written in multiplicative form
bool find_word_less_than_fixed_weight(int** generatorMatrix, int N, int K, int Q, int W, bool multiplicativeForm);

/// Searching  for a codeword with weight less than a given value of a randomly generatedlinear code for given parameters of the code
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
bool find_word_less_than_fixed_weight(int N, int K, int Q, int W);

/// Searching  for a codeword with weight less than a given value for given generator matrices, written in a file
/// @param generatorMatrixFile Name of the file, containing the generator matrices
/// @param W value of the searched weight of a codeword
/// @note The result for each linear code from the input is written in an output file "find_less_than.txt"
void find_word_less_than_fixed_weight(char* generatorMatrixFile, unsigned long long int W);

/// Searching  for a codeword with weight equal to a given value of a linear code for a given generator matrix
/// @param generatorMatrix Generator matrix, saved in 2-dimensional array
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
/// @param multiplicativeForm true if the elements of the field are written in multiplicative form
bool find_word_equal_to_fixed_weight(int** generatorMatrix, int N, int K, int Q, int W, bool multiplicativeForm);

/// Searching  for a codeword with weight equal to  a given value of a randomly generatedlinear code for given parameters of the code
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
bool find_word_equal_to_fixed_weight(int N, int K, int Q, int W);

/// Searching  for a codeword with weight equal to a given value for given generator matrices, written in a file
/// @param generatorMatrixFile Name of the file, containing the generator matrices
/// @param W value of the searched weight of a codeword
/// @note The result for each linear code from the input is written in an output file "find_equal.txt"
void find_word_equal_to_fixed_weight(char* generatorMatrixFile, unsigned long long int w_searched);

/// Calculating the number of codewords with weight less than a given value for a given generator matrix
/// @param generatorMatrix Generator matrix, saved in 2-dimensional array
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
/// @param write if true, the calculated nonproportional codewords will be written in file "Result_codewords.txt"
/// @param multiplicativeForm true if the elements of the field are written in multiplicative form
unsigned long long int calculate_number_of_words_less_than_fixed_w(int** generatorMatrix, int N, int K, int Q, int w, bool write, bool multiplicativeForm);

/// Calculating the number of codewords with weight less than a given value of a randomly generated linear code for given parameters of the code
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
/// @param write if true, the calculated nonproportional codewords will be written in file "Result_codewords.txt"
unsigned long long int calculate_number_of_words_less_than_fixed_w(int N, int K, int Q, int w, bool write);

/// Calculating the number of codewords with weight less than a given value for given generator matrices, written in a file
/// @param generatorMatrixFile Name of the file, containing the generator matrices
/// @param W value of the searched weight of a codeword
/// @param write if true, the calculated nonproportional codewords will be written in file "Result_codewords.txt"
/// @note The result for each linear code from the input is written in an output file "count_less_than.txt"
void calculate_number_of_words_less_than_fixed_w(char* generatorMatrixFile, unsigned long long int w_searched, bool write);


/// Calculate number of words with weight equal to a variable for a given generator matrix
/// @param generatorMatrix Generator matrix, saved in 2-dimensional array
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
/// @param write if true, the calculated nonproportional codewords will be written in file "Result_codewords.txt"
/// @param multiplicativeForm true if the elements of the field are written in multiplicative form 
unsigned long long int calculate_number_of_words_with_fixed_w(int** generatorMatrix, int N, int K, int Q, int w, bool write, bool multiplicativeForm);

/// Calculate number of words with weight equal to a variable of a randomly generatedlinear code for given parameters of the code
/// @param N Length of the code
/// @param K Dimension of the code
/// @param Q Number of elements in the finite field
/// @param W value of the searched weight of a codeword
/// @param write if true, the calculated nonproportional codewords will be written in file "Result_codewords.txt"
unsigned long long int calculate_number_of_words_with_fixed_w(int N, int K, int Q, int w, bool write);

/// Calculate number of words with weight equal to a variable for given generator matrices, written in a file
/// @param generatorMatrixFile Name of the file, containing the generator matrices
/// @param W value of the searched weight of a codeword
/// @param write if true, the calculated nonproportional codewords will be written in file "Result_codewords.txt"
/// @note The result for each linear code from the input is written in an output file "count_equal.txt"
void calculate_number_of_words_with_fixed_w(char* generatorMatrixFile, unsigned long long int w_searched, bool write);


#endif