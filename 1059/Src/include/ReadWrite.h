/** @file ReadWrite.h
 * @brief Function declarations for reading and writing code parameters form file and generating random generator matrices in standard form for given parameters of a code.
*/

#ifndef READ_WRITE_H
#define READ_WRITE_H
#include <iostream>
//#include "Data.h"
#include "Polynomials.h"

/// <summary>
/// Function for reading generator matrices from an open file
/// </summary>
/// <param name="fileName">Pointer to an open input stream of the input file, containing generator matrix</param>
/// <param name="n">Length of the code</param>
/// <param name="k">Dimension of the code</param>
/// <param name="q">Number of elements in the finite field</param>
/// <returns>true, if error occurs in the input (e.g. there is an character in the generator matrix instead of a number)</returns>
/// @note if the file is not open, the program end execution with an error code
bool readMatrix(FILE* fileName, int& n, int& k, int& q);

/// <summary>
/// Function for writing the weight spectrum of a linear code in to a file
/// </summary>
/// <param name="spec">Weight spectrum of a code</param>
/// <param name="N">Length of the code</param>
/// <param name="file">Name of the output file</param>
/// @note The generator matrix of the code should be written with function printMatrix printMatrix(bool form, char* file)
void printWeights(unsigned long long int* spec, int N, char* file);

/// <summary>
/// Function for writing the generator matrix in to an output file
/// </summary>
/// <param name="form">true, if the elements of the filed are written in the generator matrix in multiplicative form</param>
/// <param name="file"> Name of the output file</param>
/// @note The generator matrix is read from the global variable @ref matrix
void printMatrix(bool form, char* file);

/// <summary>
/// Generating random generator matrices of a codes with given parameters
/// </summary>
/// <param name="n">Length of the code</param>
/// <param name="k">Dimension of the code</param>
/// <param name="q">Number of elements in the finite field</param>
/// <param name="num">Number of matrices to be generated </param>
/// @note The generator matrices are written in output file **EXAM**. If just one matrix is to be generated, it is saved in global variable  @ref matrix
void randomgenf(int n, int k, int q, int num);

/// <summary>
/// Generating a random generator matrix of a code for given parameters
/// </summary>
/// <param name="n">Length of the code</param>
/// <param name="k">Dimension of the code</param>
/// <param name="q">Number of elements in the finite field</param>
/// <param name="generatorMatrix">A pointer to a dynamic 2D array that will contain the generator matrices</param>
/// <param name="multiplicativeForm">true if the generator matrix is to be saved in multiplicative form</param>
/// @note The generator matrix are written in output file **EXAM**
void randomgenf(int n, int k, int q, int** generatorMatrix, bool multiplicativeForm);

/// <summary>
/// Function for writing an element of the field in multiplicative form
/// </summary>
/// <param name="dec">Decimal representation of the element</param>
/// <param name="write">Pointer to an open stream for writing  </param>
/// @note If an incorrect decimal is given, the function will end the program with an error code and an error message will be written in file **error.txt**
void write_multpl(int dec,  FILE* write); /// function for writing an element of the field in multiplicative form

/// <summary>
/// Function for writing an element of the field in multiplicative form in magma style
/// </summary>
/// <param name="dec">Decimal representation of the element</param>
/// <param name="write">Pointer to an open stream for writing  </param>
/// @note If an incorrect decimal is given, the function will end the program with an error code and an error message will be written in file **error.txt**
void write_multpl_magma(int dec, FILE* write); ///function for writing an element of the field in multiplicative form in magma style

#endif // READ_WRITE_H

