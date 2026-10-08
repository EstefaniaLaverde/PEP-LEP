/** @file TestDriver.h
* @brief Declaration of a function for correctness and runtime testing. 
*/


#ifndef TEST_DRIVER_H
#define TEST_DRIVER_H
#include "LinCodeWeightInv.h"
#include "Data.h"

/// <summary>
/// This function gives a simple interface designed to test all end user functionalities of the presented library with different instruction sets.
/// The interface lets the user to chose with which instruction sets the computations will be executed.
/// If ARM architecture is detected, appropriate message is displayed.
/// </summary>
/// @note The results (average execution times, total computational time) of the calculations are written in file **Results**. The program ends with error message if there are errors in the computations. Error message is written in file **error.txt**
void test_drive();

/// <summary>
/// Function for testing functionalities with data from the corresponding manuscript with registers with 128-bit registers.
/// </summary>
/// <param name="mode">Shows which function will be tested. 
/// Possible values are 1, 2, 3, 4, 5, 6 that correspond  to the functions  for calculating weight distribution, minimum distance, searching for codeword with weight equal given value or less than given value,
/// counting the codewords with weight equal to or less than given value, respectively.
/// </param>

void test_driveSSE(int mode);
/// <summary>
/// Function for testing functionalities with data from the corresponding manuscript with registers with 256-bit registers.
/// </summary>
/// <param name="mode">Shows which function will be tested. 
/// Possible values are 1, 2, 3, 4, 5, 6 that correspond  to the functions  for calculating weight distribution, minimum distance, searching for codeword with weight equal given value or less than given value,
/// counting the codewords with weight equal to or less than given value, respectively.
/// </param>
void test_driveAVX(int mode);

/// <summary>
/// Function for testing functionalities with data from the corresponding manuscript with registers with 512-bit registers.
/// </summary>
/// <param name="mode">Shows which function will be tested. 
/// Possible values are 1, 2, 3, 4, 5, 6 that correspond  to the functions  for calculating weight distribution, minimum distance, searching for codeword with weight equal given value or less than given value,
/// counting the codewords with weight equal to or less than given value, respectively.
/// </param>
void test_driveAVX512(int mode);

/// <summary>
/// Function for testing functionalities with data from the corresponding manuscript with registers without vectorization.
/// </summary>
/// <param name="mode">Shows which function will be tested. 
/// Possible values are 1, 2, 3, 4, 5, 6 that correspond  to the functions  for calculating weight distribution, minimum distance, searching for codeword with weight equal given value or less than given value,
/// counting the codewords with weight equal to or less than given value, respectively.
/// </param>
void test_driveScalar(int mode);


#endif

