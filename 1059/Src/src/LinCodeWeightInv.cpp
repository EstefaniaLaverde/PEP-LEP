
#include"LinCodeWeightInv.h"
#include "ReadWrite.h"
#include "Polynomials.h"
#include "Data.h"

#include <iostream>
#include <fstream>
#include <time.h>


#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include "lib512.h"
#include "lib256.h"
#include "lib128.h"
#elif defined (__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#include <cpuid.h>
#include "lib512.h"
#include "lib256.h"
#include "lib128.h"
#elif defined(__GNUC__)
#include "lib_neon.h"
#endif


using namespace std;
 // set to true to force specific instruction set (may result in problems if not available)
int instructionSet = 6; // and set instr to one of the following values

/* 1->SSE
*  2 -> SSE2
*  3 -> SSE3
*  4 -> SSSE3
*  5 -> SSE 4.1 // blend
*  6 -> SSE 4.2
*  7 -> AVX
*
* cpuid (7, abcd)
*  8 -> AVX2
*  9 -> AVX512 F
*  10 -> AVX512 vl

*  11 -> AVX 512 popcnt + vl
*/
//------------ MSC  ------------//

int n = 0, k = 0, q = 0;
unsigned long long int d_fin = N_FIX*8+2;
FILE* write;
bool multipl = false;
unsigned long long int count_mat = 0;

//constants for evaluating if the code will be able to fit in static memory
// comments show current value
static const int K_64 = K_FIX / 6; // 6
static const int N_64 = N_FIX; // 3072

static const int K_32 = K_FIX / 5; // 7
static const int N_32 = N_FIX; // 3072

static const int K_16 = K_FIX / 4; // 9
static const int N_16 = N_FIX; // 3072

static const int K_8 = K_FIX / 3; // 12
static const int N_8 = N_FIX; // 3072

static const int K_4 = K_FIX / 2; // 18
static const int N_4 = N_FIX * 4; //12288

static const int K_2 = K_FIX; // 36
static const int N_2 = N_FIX * 8; //24576

static const int K_3 = K_FIX ; // 36
static const int N_3 = N_FIX * 4; // 12288

static const int K_9 = K_FIX / 2; // 18
static const int N_9 = N_FIX * 2; //6144

static const int K_27 = K_FIX / 3; // 12
static const int N_27 = N_CH3 * 8; // 512*8 = 4096

static const int K_P_cf = K_FIX / 4; // 9

// NOTE (patched, Aug 2026): the original table here was declared
// `bool is_prime_power[65]` covering only q = 0..64. Every "default"
// switch case in init_zero() below (i.e. any q not one of the explicitly
// hardcoded 2,3,4,8,9,16,25,27,32,49,64) reads is_prime_power[q] with NO
// bounds check, so any q > 64 - such as q = 127, used throughout the
// PEP-LEP thesis instances - was an out-of-bounds array read (undefined
// behavior), not a real "is q a prime power" test. Extended to size 128
// (indices 0..127) with the same true/false convention as the original
// table (is_prime_power(n) := n == p^k for a single prime p, k >= 1;
// indices 0 and 1 kept as `true`, matching the original table's own
// convention rather than strict number theory). Verified by direct
// computation to reproduce the original table exactly on indices 0..64
// before extending it.
bool is_prime_power[128] = {
	true, true, true, true, true, true, false, true, true, true,
	false, true, false, true, false, false, true, true, false, true,
	false, false, false, true, false, true, false, true, false, true,
	false, true, true, false, false, false, false, true, false, false,
	false, true, false, true, false, false, false, true, false, true,
	false, false, false, true, false, false, false, false, false, true,
	false, true, false, false, true, false, false, true, false, false,
	false, true, false, true, false, false, false, false, false, true,
	false, true, false, true, false, false, false, false, false, true,
	false, false, false, false, false, false, false, true, false, false,
	false, true, false, true, false, false, false, true, false, true,
	false, false, false, true, false, false, false, false, false, false,
	false, true, false, false, false, true, false, true
};

// --------------------------- detecting the instruction set --------------------------//
void detect() {
#if defined(_MSC_VER)

	int abcd[4] = { 0,0,0,0 };
	__cpuid(abcd, 7);
	if ((abcd[1] & (1 << 16)) && (abcd[1] & (1 << 30))) {
		instructionSet = 9;
		//printf("AVX512 F\n");
		if ((abcd[2] & (1 << 14))) { // && (abcd[1] & (1 << 31)) //vl
			//printf("AVX 512 popcnt + vl\n");
			instructionSet = 11;
			return ;
		}
		return;
	}

	if (abcd[1] & (1 << 5)) {
		instructionSet = 8;
		//printf("AVX2\n");
		return ;
	}

	__cpuid(abcd, 1);
	if (abcd[2] & (1 << 19)) {
		//printf("SSE4.1\n");
		instructionSet = 5;
		return;
	}
	if (abcd[3] & (1 << 26)) {
		//printf("SSE2\n");
		instructionSet = 2;
		return ;
	}
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
	unsigned int abcd[4] = { 0,0,0,0 };

	__get_cpuid(1, &abcd[0], &abcd[1], &abcd[2], &abcd[3]);
	__get_cpuid_count(7,0, &abcd[0], &abcd[1], &abcd[2], &abcd[3]);

	if ((abcd[1] & (1 << 16)) && (abcd[1] & (1 << 30))) {
		instructionSet = 9;
		if ((abcd[2] & (1 << 14)) ) { // && (abcd[1] & (1 << 31)) //vl
			instructionSet = 11;
			return ;
		}
		return;
	}

	if (abcd[1] & (1 << 5)) {
		instructionSet = 8;
		return ;
	}

	__get_cpuid(1, &abcd[0], &abcd[1], &abcd[2], &abcd[3]);
	if (abcd[2] & (1 << 19)) {
		instructionSet = 5;
		return;
	}
	if (abcd[3] & (1 << 26)) {
		instructionSet = 2;
		return ;
	}
#endif
}

//--------------------- initializing the matrices ---------------------//
bool init_zero() {
	//printf("n = %d k = %d q = %d\n\n", n, k, q);

	bool enough_mem = false;
	switch (q)
	{
	case 2:
		if((k <= K_2 && n <= N_2)) enough_mem = true;
		break;
	case 3:
		if((k <= K_3 && n <= N_3)) enough_mem = true;
		break;
	case 4:
		if((k <= K_4 && n <= N_4)) enough_mem = true;
		break;
	case 8:
		if((k <= K_8 && n <= N_8)) enough_mem = true;
		break;
	case 9:
		if((k <= K_9 && n <= N_9)) enough_mem = true;
		break;
	case 16:
		if((k <= K_16 && n <= N_16)) enough_mem = true;
		break;
	case 25:
		if((k <= K_P_cf && n <= N_FIX)) enough_mem = true;
		break;
	case 27:
		if((k <= K_27 && n <= N_27)) enough_mem = true;
		break;
	case 32:
		if((k <= K_32 && n <= N_32)) enough_mem = true;
		break;
	case 49:
		if ((k <= K_P_cf && n <= N_FIX)) enough_mem = true;
		break;
	case 64:
		if((k <= K_64 && n <= N_64)) enough_mem = true;
		break;
	default:
		if((is_prime_power[q]) && (k <= K_P && n <= N_P)) enough_mem = true;
		else enough_mem = false;
		break;
	}


	if (enough_mem) {
		dmat_short_new(matrix, K_FIX + 1, N_FIX * 8, q);
		dmat_short_new(matrixH, K_FIX + 1, N_FIX * 8, q);

		// same as in lib128 !!!!!!!!!!!!!!! //

		if ((q == 2) || (q == 4)) {
			dmat_short_new(bits, K_GF2, N_GF2, q);
			//dmat_new(bitsCharCF, 60, 512, q);
		}
		else if ((q == 8) || (q == 16) || (q == 32) || (q == 64)) {
			//dmat_short_new(bits, 60, 32768, q);
			dmat_new(bitsCharCF, K_GF2, N_CH2, q);
		}
		else if ((q == 3) || (q == 9) || (q == 27)) {
			dmat_short_new(bits, K_CH3, N_CH3, q);
			//dmat_new(bitsCharCF, 30, 1024, q);
		}

		else if (is_prime_power[q] || (q == 25) || (q == 49)) {
			//dmat_short_new(bits, K_P, N_P, q);
			dmat_new(bitsCharCF, K_P, N_P, q);
		}
		else {
			return false;
		}



		for (int i = 0; i <= 8 * N_FIX; i++) {
			weights[i] = 0;
		}

		for (int i = 0; i < matrix.k; i++) {
			for (int j = 0; j < matrix.n; j++) {
				matrix.a[i][j] = 0;
				matrixH.a[i][j] = 0;
			}
		}
		if (bits.mem != 0) {
			for (int i = 0; i < bits.k; i++) {
				for (int j = 0; j < bits.n; j++) {
					bits.a[i][j] = 0;
				}
			}
		}
		if (bitsCharCF.mem != 0) {
			for (int i = 0; i < bitsCharCF.k; i++) {
				for (int j = 0; j < bitsCharCF.n; j++) {
					bitsCharCF.a[i][j] = 0;
				}
			}
		}
		return true;
	}

	else {
		return false;
	}
}

bool init() {// generates multiplicaton table and transforms the input matrix to appropriate from for the current q

		maketable(q);
		//da se proveri multiplicativen zapis ili desetichen
		// funkciqta da e bool
		// priema parametar
		int temp = 0;
		if (multipl) {
			for (int i = 0; i < matrix.k; i++) {
				for (int j = 0; j < matrix.n; j++) {
					temp =  multipl_to_dec((int)matrix.a[i][j]);
					if (temp<0)
					{
						printf("Error in input matrix (multiplicative form of elements)\n\n");
						return false;
					}
					matrix.a[i][j] = temp;
				}
			}
		}

			switch (q) {
			case 2:
				num_to_coef_gf2(k, n, 1);
				break;
			case 3:
				num_to_coef_gf3(k, n, 1);
				break;
			case 4:
				multiplyGF4(k, n);
				num_to_coef_gf4(k, n, 2);
				//num_to_coef_char2(matrix, k, n, 2, bitsCharCF);
				break;
			case 8:
				multiplyGF8(k, n);
				//num_to_coef_gf8(matrix, k, n, 3, bits);
				num_to_coef_char2(k, n, 3);
				break;
			case 9:
				multiplyGF9(k, n);
				num_to_coef_gf9(k, n, 1);
				break;
			case 16:
				multiplyGF16(k, n);
				num_to_coef_char2(k, n, 4);
				//num_to_coef_gf16(matrix, k, n, 4, bits);
				break;
			case 25:
				multiplyGF25(k, n);
				num_to_coef_char25(k, n, 2);
				break;
			case 32:
				multiplyGF32(k, n);
				num_to_coef_char2(k, n, 5);
				//num_to_coef_gf32(matrix, k, n, 5, bits);
				break;
			case 27:
				multiplyGF27(k, n);
				num_to_coef_gf27(k, n, 1);
				break;
			case 49:
				multiplyGF49(k, n);
				num_to_coef_char49(k, n, 2);
				break;
			case 64:
				multiplyGF64(k, n);
				num_to_coef_char2(k, n, 6);
				//num_to_coef_gf64(matrix, k, n, 6, bits);
				break;

			default:
				if (is_prime_power[q]) {
					num_to_coef_char2(k, n, 1);
				}
				break;
			}
			return true;
}

// ---------------------------------- table multiplication functions -------------------------//
bool flag = false;
void linear_combinations(int rec, int h) {
	int qf = q;
	if (h == 1) { qf = 2; }
	else { qf = q; }
	for (int i = h; i <= k; i++) {
		for (int q1 = 1; q1 < qf; q1++) {
			int t = add_vector( n, rec , i, q1);
			if (t > 0)	weights[t]++;
			if (rec < k) { linear_combinations(rec + 1, i + 1); }
		}
	}
}
void linear_combinations_equal(int rec, int h, int d_fin) {
	if (!flag) {
		int qf = 2;
		if (h == 1) { qf = 2; }
		else { qf = q; }
		for (int i = h; i <= k; i++) {
			for (int q1 = 1; q1 < qf; q1++) {
				int t = add_vector(n, rec, i, q1);
				if (t == d_fin) {
					//printf("FOUND (for testing)!\n\n");
					flag = true;
				}
				if (t > 0)	weights[t]++;
				if (rec < k) { linear_combinations_equal(rec + 1, i + 1, d_fin); }
			}
		}
	}

}
void linear_combinations_less_than(int rec, int h, int d_fin) {
	if (!flag) {
		int qf = 2;
		if (h == 1) { qf = 2; }
		else { qf = q; }
		for (int i = h; i <= k; i++) {
			for (int q1 = 1; q1 < qf; q1++) {
				int t = add_vector( n, rec, i, q1);
				if (t < d_fin) {
					//printf("FOUND (for testing)\n");
					flag = true;
				}
				if (t > 0)	weights[t]++;
				if (rec < k) { linear_combinations_less_than(rec + 1, i + 1, d_fin); }
			}
		}
	}
}
void linear_combinations_less_than_count(int rec, int h, int d_fin) {

	int qf = 2;
	if (h == 1) { qf = 2; }
	else { qf = q; }
	for (int i = h; i <= k; i++) {
		for (int q1 = 1; q1 < qf; q1++) {
			int t = add_vector( n, rec, i, q1);
			if (write!=NULL) {
				if (t < d_fin) {
					for (int j = 0; j < n; j++) {
						if (multipl) {
							write_multpl((int)matrixH.a[rec][j],write);
						}else{
							fprintf(write, "%llu, ", matrixH.a[rec][j]);
						}

					}
					fprintf(write, "\n");
				}

			}
			if (t > 0) { weights[t]++; }
			if (rec < k) { linear_combinations_less_than_count(rec + 1, i + 1, d_fin); }
		}
	}
}
void linear_combinations_equal_count(int rec, int h, int d_fin) {

	int qf = 2;
	if (h == 1) { qf = 2; }
	else { qf = q; }
	for (int i = h; i <= k; i++) {
		for (int q1 = 1; q1 < qf; q1++) {
			int t = add_vector(n, rec, i, q1);
			if (write!=NULL) {
				if (t == d_fin) {
					for (int j = 0; j < n; j++) {
						if (multipl) {
							write_multpl((int)matrixH.a[rec][j], write);

						}
						else {
							fprintf(write, "%llu, ", matrixH.a[rec][j]);
						}
					}
					fprintf(write, "\n");
				}
			}
			if (t > 0) { weights[t]++; }
			if (rec < k) { linear_combinations_equal_count(rec + 1, i + 1, d_fin); }
		}
	}
}

//pointer functions -- needed for future inegration of 256 and 512-bit registers

void (*calculateWeightGF2) (dynamic_mat_short&, int, int);
void (*calculateWeightCH3) (dynamic_mat_short&, int, int, int);
void (*calculateWeightCH2) (dynamic_mat_short&, int, int, int);
void (*calculateWeightBytes) (dmat_type&, int, int, int, int);


bool (*calculateWeightGF2_less_than) (dynamic_mat_short&, int, int, int);
bool (*calculateWeightCH3_less_than) (dynamic_mat_short&, int, int, int, int);
bool (*calculateWeightCH2_less_than) (dynamic_mat_short&, int, int, int, int);
bool (*calculateWeightBytes_less_than) (dmat_type&, int, int, int, int, int);

bool (*calculateWeightGF2_equal) (dynamic_mat_short&, int, int, int);
bool (*calculateWeightCH3_equal) (dynamic_mat_short&, int, int, int, int);
bool (*calculateWeightCH2_equal) (dynamic_mat_short&, int, int, int,  int);
bool (*calculateWeightBytes_equal) (dmat_type&, int, int, int, int,  int);

unsigned long long int (*count_number_of_wordsGF2_less_than) (dynamic_mat_short&, int, int, int, bool);
unsigned long long int (*count_number_of_wordsCH3_less_than) (dynamic_mat_short&, int, int, int,int,bool);
unsigned long long int (*count_number_of_wordsCH2_less_than) (dynamic_mat_short&, int, int, int, int,bool);
unsigned long long int (*count_number_of_wordsBytes_less_than) (dmat_type&, int, int, int, int, int,bool);

unsigned long long int (*count_number_of_wordsGF2_equal) (dynamic_mat_short&, int, int,  int,bool);
unsigned long long int (*count_number_of_wordsCH3_equal) (dynamic_mat_short&, int, int, int,  int,bool);
unsigned long long int (*count_number_of_wordsCH2_equal) (dynamic_mat_short&, int, int, int,  int,bool);
unsigned long long int (*count_number_of_wordsBytes_equal) (dmat_type&, int, int, int, int, int,bool);


// end  pointer functions

 // -------------------------- dispatch function for detecting register --------------------------------------//
void dispatch_try(int instructionSet) {
//#if defined(__ARM_NEON__)
           // printf("neon\n");

//#elif defined (_MSC_VER)||((__AVX512F__ )&&(__AVX512BW__))
//#elif defined (__AVX512F__ )&&(__AVX512BW__)
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
if (instructionSet >= 9) {
		//printf("AVX512\n");
		calculateWeightGF2 = &calculateWeightGF2_512;
		calculateWeightCH3 = &calculateWeightCH3_512;
		calculateWeightCH2 = &calculateWeightCH2_512;
		calculateWeightBytes = &calculateWeightBytes_512;

		calculateWeightGF2_less_than = &calculateWeightGF2_512_less_than;
		calculateWeightCH3_less_than = &calculateWeightCH3_512_less_than;
		calculateWeightCH2_less_than = &calculateWeightCH2_512_less_than;
		calculateWeightBytes_less_than = &calculateWeightBytes_512_less_than;

		calculateWeightGF2_equal = &calculateWeightGF2_512_equal;
		calculateWeightCH3_equal = &calculateWeightCH3_512_equal;
		calculateWeightCH2_equal = &calculateWeightCH2_512_equal;
		calculateWeightBytes_equal = &calculateWeightBytes_512_equal;

		count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_512_less_than;
		count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_512_less_than;
		count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_512_less_than;
		count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_512_less_than;

		count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_512_equal;
		count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_512_equal;
		count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_512_equal;
		count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_512_equal;
		}
	else if (instructionSet >= 8) {
		//printf("AVX2\n");
		calculateWeightGF2 = &calculateWeightGF2_256;
		calculateWeightCH3 = &calculateWeightCH3_256;
		calculateWeightCH2 = &calculateWeightCH2_256;
		calculateWeightBytes = &calculateWeightBytes_256;

		calculateWeightGF2_less_than = &calculateWeightGF2_256_less_than;
		calculateWeightCH3_less_than = &calculateWeightCH3_256_less_than;
		calculateWeightCH2_less_than = &calculateWeightCH2_256_less_than;
		calculateWeightBytes_less_than = &calculateWeightBytes_256_less_than;

		calculateWeightGF2_equal = &calculateWeightGF2_256_equal;
		calculateWeightCH3_equal = &calculateWeightCH3_256_equal;
		calculateWeightCH2_equal = &calculateWeightCH2_256_equal;
		calculateWeightBytes_equal = &calculateWeightBytes_256_equal;

		count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_256_less_than;
		count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_256_less_than;
		count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_256_less_than;
		count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_256_less_than;

		count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_256_equal;
		count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_256_equal;
		count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_256_equal;
		count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_256_equal;

	}
	else {//if (instructionSet >= 5) {
		//printf("SSE\n");
		calculateWeightGF2 = &calculateWeightGF2_128;
		calculateWeightCH3 = &calculateWeightCH3_128;
		calculateWeightCH2 = &calculateWeightCH2_128;
		calculateWeightBytes = &calculateWeightBytes_128;

		calculateWeightGF2_less_than = &calculateWeightGF2_128_less_than;
		calculateWeightCH3_less_than = &calculateWeightCH3_128_less_than;
		calculateWeightCH2_less_than = &calculateWeightCH2_128_less_than;
		calculateWeightBytes_less_than = &calculateWeightBytes_128_less_than;

		calculateWeightGF2_equal = &calculateWeightGF2_128_equal;
		calculateWeightCH3_equal = &calculateWeightCH3_128_equal;
		calculateWeightCH2_equal = &calculateWeightCH2_128_equal;
		calculateWeightBytes_equal = &calculateWeightBytes_128_equal;

		count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_128_less_than;
		count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_128_less_than;
		count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_128_less_than;
		count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_128_less_than;

		count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_128_equal;
		count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_128_equal;
		count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_128_equal;
		count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_128_equal;
	}
#else
			calculateWeightGF2 = &calculateWeightGF2_neon;
			calculateWeightCH3 = &calculateWeightCH3_neon;
			calculateWeightCH2 = &calculateWeightCH2_neon;
			calculateWeightBytes = &calculateWeightBytes_neon;

			calculateWeightGF2_less_than = &calculateWeightGF2_neon_less_than;
			calculateWeightCH3_less_than = &calculateWeightCH3_neon_less_than;
			calculateWeightCH2_less_than = &calculateWeightCH2_neon_less_than;
			calculateWeightBytes_less_than = &calculateWeightBytes_neon_less_than;

			calculateWeightGF2_equal = &calculateWeightGF2_neon_equal;
			calculateWeightCH3_equal = &calculateWeightCH3_neon_equal;
			calculateWeightCH2_equal = &calculateWeightCH2_neon_equal;
			calculateWeightBytes_equal = &calculateWeightBytes_neon_equal;

			count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_neon_less_than;
			count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_neon_less_than;
			count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_neon_less_than;
			count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_neon_less_than;

			count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_neon_equal;
			count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_neon_equal;
			count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_neon_equal;
			count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_neon_equal;
/*#elif defined (__AVX2__)
	if (instructionSet >= 8) {
		//printf("AVX2\n");
		calculateWeightGF2 = &calculateWeightGF2_256;
		calculateWeightCH3 = &calculateWeightCH3_256;
		calculateWeightCH2 = &calculateWeightCH2_256;
		calculateWeightBytes = &calculateWeightBytes_256;

		calculateWeightGF2_less_than = &calculateWeightGF2_256_less_than;
		calculateWeightCH3_less_than = &calculateWeightCH3_256_less_than;
		calculateWeightCH2_less_than = &calculateWeightCH2_256_less_than;
		calculateWeightBytes_less_than = &calculateWeightBytes_256_less_than;

		calculateWeightGF2_equal = &calculateWeightGF2_256_equal;
		calculateWeightCH3_equal = &calculateWeightCH3_256_equal;
		calculateWeightCH2_equal = &calculateWeightCH2_256_equal;
		calculateWeightBytes_equal = &calculateWeightBytes_256_equal;

		count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_256_less_than;
		count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_256_less_than;
		count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_256_less_than;
		count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_256_less_than;

		count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_256_equal;
		count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_256_equal;
		count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_256_equal;
		count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_256_equal;
	}
	else if (instructionSet >= 5) {
		//printf("SSE\n");
		calculateWeightGF2 = &calculateWeightGF2_128;
		calculateWeightCH3 = &calculateWeightCH3_128;
		calculateWeightCH2 = &calculateWeightCH2_128;
		calculateWeightBytes = &calculateWeightBytes_128;

		calculateWeightGF2_less_than = &calculateWeightGF2_128_less_than;
		calculateWeightCH3_less_than = &calculateWeightCH3_128_less_than;
		calculateWeightCH2_less_than = &calculateWeightCH2_128_less_than;
		calculateWeightBytes_less_than = &calculateWeightBytes_128_less_than;

		calculateWeightGF2_equal = &calculateWeightGF2_128_equal;
		calculateWeightCH3_equal = &calculateWeightCH3_128_equal;
		calculateWeightCH2_equal = &calculateWeightCH2_128_equal;
		calculateWeightBytes_equal = &calculateWeightBytes_128_equal;

		count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_128_less_than;
		count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_128_less_than;
		count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_128_less_than;
		count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_128_less_than;

		count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_128_equal;
		count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_128_equal;
		count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_128_equal;
		count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_128_equal;
	}
#elif defined (__SSE4_1__)//||(__SSE2__)
	if (instructionSet >= 5) {
			//printf("SSE\n");
			calculateWeightGF2 = &calculateWeightGF2_128;
			calculateWeightCH3 = &calculateWeightCH3_128;
			calculateWeightCH2 = &calculateWeightCH2_128;
			calculateWeightBytes = &calculateWeightBytes_128;

			calculateWeightGF2_less_than = &calculateWeightGF2_128_less_than;
			calculateWeightCH3_less_than = &calculateWeightCH3_128_less_than;
			calculateWeightCH2_less_than = &calculateWeightCH2_128_less_than;
			calculateWeightBytes_less_than = &calculateWeightBytes_128_less_than;

			calculateWeightGF2_equal = &calculateWeightGF2_128_equal;
			calculateWeightCH3_equal = &calculateWeightCH3_128_equal;
			calculateWeightCH2_equal = &calculateWeightCH2_128_equal;
			calculateWeightBytes_equal = &calculateWeightBytes_128_equal;

			count_number_of_wordsBytes_less_than = &calculateNumberOfWordsBytes_128_less_than;
			count_number_of_wordsCH2_less_than = &calculateNumberOfWordsCH2_128_less_than;
			count_number_of_wordsGF2_less_than = &calculateNumberOfWordsGF2_128_less_than;
			count_number_of_wordsCH3_less_than = &calculateNumberOfWordsCH3_128_less_than;

			count_number_of_wordsBytes_equal = &calculateNumberOfWordsBytes_128_equal;
			count_number_of_wordsCH2_equal = &calculateNumberOfWordsCH2_128_equal;
			count_number_of_wordsGF2_equal = &calculateNumberOfWordsGF2_128_equal;
			count_number_of_wordsCH3_equal = &calculateNumberOfWordsCH3_128_equal;
	}*/
#endif
}

// ---------- Functions for chosing register and optimized function for calculations according to q -------------------------//

void calculateWeightDistrib() {
#if FORCE_INSTR==0
	detect();
#else
	instructionSet = FORCE_INSTR;
#endif
	if (test >= 0) { instructionSet = test; }
	dispatch_try(instructionSet);

		if (instructionSet >= 5) { // blend SSE4.1
			switch (q)
			{
			case 2:
				calculateWeightGF2(bits, n, k);
				break;
			case 3:
				calculateWeightCH3(bits, n, k, 1);
				break;
			case 4:
				//calculateWeightBytes(bitsCharCF, n, k, 2, 2);
				calculateWeightCH2(bits, n, k, 2);
				break;
			case 8:
				calculateWeightBytes(bitsCharCF, n, k, 3, 2);
				//calculateWeightCH2(bits, n, k, 3); // bitwise representation can be used instead of byte representation
				break;
			case 9:
				calculateWeightCH3(bits, n, k, 2);
				break;
			case 16:
				calculateWeightBytes(bitsCharCF, n, k, 4, 2);
				//calculateWeightCH2(bits, n, k, 4);// bitwise representation can be used instead of byte representation
				break;
			case 25:
				calculateWeightBytes(bitsCharCF, n, k, 2, 5);
				break;
			case 32:
				calculateWeightBytes(bitsCharCF, n, k, 5, 2);
				//calculateWeightCH2(bits, n, k, 5);// bitwise representation can be used instead of byte representation
				break;
			case 27:
				calculateWeightCH3(bits, n, k, 3);
				break;
			case 49:
				calculateWeightBytes(bitsCharCF, n, k, 2, 7);
				break;
			case 64:
				//calculateWeightCH2(bits, n, k, 6);// bitwise representation can be used instead of byte representation
				calculateWeightBytes(bitsCharCF, n, k, 6, 2);
				break;
			default:
				if (is_prime_power[q]) {
					calculateWeightBytes(bitsCharCF, n, k, 1, q);
				}
				else {
					printf("not prime <64 \n\n");
				}
				break;
			}
		}
		else {
			//printf("Calculation using look up tables!\n\n");
			linear_combinations(1, 1);

			for (int i = 0; i <= n; i++) {
				weights[i] = weights[i]*(q-1);
			}

		}



}

bool find_word_less_than_fixed_w(int d_fin) {
	bool res = false;
	flag = false;
#if FORCE_INSTR==0
	detect();
#else
	instructionSet = FORCE_INSTR;
#endif
	if (test >= 0) { instructionSet = test; }
	//if (!forceISTR) { detect(); }
	dispatch_try(instructionSet);
		if (instructionSet >= 5) { // blend SSE4.1
			switch (q)
			{
			case 2:
				res = calculateWeightGF2_less_than(bits, n, k, d_fin);
				break;
			case 3:
				res = calculateWeightCH3_less_than(bits, n, k, 1, d_fin);
				break;
			case 4:
				//calculateWeightBytes(bitsCharCF, n, k, 2, 2);
				res = calculateWeightCH2_less_than(bits, n, k, 2, d_fin);
				break;
			case 8:
				res = calculateWeightBytes_less_than(bitsCharCF, n, k, 3, 2, d_fin);
				//calculateWeightCH2(bits, n, k, 3);// bitwise representation can be used instead of byte representation
				break;
			case 9:
				res = calculateWeightCH3_less_than(bits, n, k, 2, d_fin);
				break;
			case 16:
				res = calculateWeightBytes_less_than(bitsCharCF, n, k, 4, 2, d_fin);
				//calculateWeightCH2(bits, n, k, 4);
				break;
			case 25:
				res = calculateWeightBytes_less_than(bitsCharCF, n, k, 2, 5, d_fin);
				break;
			case 32:
				res = calculateWeightBytes_less_than(bitsCharCF, n, k, 5, 2, d_fin);
				//calculateWeightCH2(bits, n, k, 5);
				break;
			case 27:
				res = calculateWeightCH3_less_than(bits, n, k, 3, d_fin);
				break;
			case 49:
				res = calculateWeightBytes_less_than(bitsCharCF, n, k, 2, 7, d_fin);
				break;
			case 64:
				//calculateWeightCH2(bits, n, k, 6);
				res = calculateWeightBytes_less_than(bitsCharCF, n, k, 6, 2, d_fin);
				break;
			default:
				if (is_prime_power[q]) {
					res = calculateWeightBytes_less_than(bitsCharCF, n, k, 1, q, d_fin);
				}
				else {
					printf("not prime <64 \n\n");
				}
				break;
			}
		}
		else {
			//printf("Calculation using lookup tables!\n\n");
			linear_combinations_less_than(1, 1, d_fin);

			res = flag;

			for (int i = 0; i <= n; i++) {
				weights[i] = weights[i] * (q - 1);
			}

		}
	return res;
}



bool find_word_equal_fixed_w(int d_fin) {
	bool res = false;
	flag = false;
#if FORCE_INSTR==0
	detect();
#else
	instructionSet = FORCE_INSTR;
#endif
	if (test >= 0) { instructionSet = test; }
	//if (!forceISTR) { detect(); }
	dispatch_try(instructionSet);
		clock_t begin, end;
		if (instructionSet >= 5) { // blend SSE4.1
			switch (q)
			{
			case 2:
				res = calculateWeightGF2_equal(bits, n, k, d_fin);
				break;
			case 3:
				res = calculateWeightCH3_equal(bits, n, k, 1, d_fin);
				break;
			case 4:
				//calculateWeightBytes(bitsCharCF, n, k, 2, 2);
				res = calculateWeightCH2_equal(bits, n, k, 2, d_fin);
				break;
			case 8:
				res = calculateWeightBytes_equal(bitsCharCF, n, k, 3, 2, d_fin);
				//calculateWeightCH2(bits, n, k, 3);// bitwise representation can be used instead of byte representation
				break;
			case 9:
				res = calculateWeightCH3_equal(bits, n, k, 2, d_fin);
				break;
			case 16:
				res = calculateWeightBytes_equal(bitsCharCF, n, k, 4, 2, d_fin);
				//calculateWeightCH2(bits, n, k, 4);
				break;
			case 25:
				res = calculateWeightBytes_equal(bitsCharCF, n, k, 2, 5, d_fin);
				break;
			case 32:
				res = calculateWeightBytes_equal(bitsCharCF, n, k, 5, 2, d_fin);
				//calculateWeightCH2(bits, n, k, 5);
				break;
			case 27:
				res = calculateWeightCH3_equal(bits, n, k, 3, d_fin);
				break;
			case 49:
				res = calculateWeightBytes_equal(bitsCharCF, n, k, 2, 7, d_fin);
				break;
			case 64:
				//calculateWeightCH2(bits, n, k, 6);
				res = calculateWeightBytes_equal(bitsCharCF, n, k, 6, 2, d_fin);
				break;
			default:
				if (is_prime_power[q]) {
					res = calculateWeightBytes_equal(bitsCharCF, n, k, 1, q, d_fin);
				}
				else {
					printf("not prime <64 \n\n");
				}
				break;
			}
		}
		else {
			//printf("Calculation using lookup tables!\n\n");
			linear_combinations_equal(1, 1, d_fin);
			res = flag;

			for (int i = 0; i <= n; i++) {
				weights[i] = weights[i] * (q - 1);
			}

		}

	return res;
}


unsigned long long int  count_word_less_than_fixed_w(int d_fin, bool multiplicativeForm) {
#if FORCE_INSTR==0
	detect();
#else
	instructionSet = FORCE_INSTR;
#endif
	if (test >= 0) { instructionSet = test; }
	//if (!forceISTR) { detect(); }
	unsigned long long int res = 0;

	dispatch_try(instructionSet);
		if (instructionSet >= 5) { // blend SSE4.1
			switch (q)
			{
			case 2:
				res = count_number_of_wordsGF2_less_than(bits, n, k, d_fin, multiplicativeForm);
				break;
			case 3:
				res = count_number_of_wordsCH3_less_than(bits, n, k, 1, d_fin, multiplicativeForm);
				break;
			case 4:
				//calculateWeightBytes(bitsCharCF, n, k, 2, 2);
				res = count_number_of_wordsCH2_less_than(bits, n, k, 2,  d_fin, multiplicativeForm);
				break;
			case 8:
				res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 3, 2, d_fin, multiplicativeForm);
				//calculateWeightCH2(bits, n, k, 3);
				break;
			case 9:
				res = count_number_of_wordsCH3_less_than(bits, n, k, 2, d_fin, multiplicativeForm);
				break;
			case 16:
				res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 4, 2, d_fin, multiplicativeForm);
				//calculateWeightCH2(bits, n, k, 4);
				break;
			case 25:
				res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 2, 5, d_fin, multiplicativeForm);
				break;
			case 32:
				res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 5, 2, d_fin, multiplicativeForm);
				//calculateWeightCH2(bits, n, k, 5);
				break;
			case 27:
				res = count_number_of_wordsCH3_less_than(bits, n, k, 3, d_fin, multiplicativeForm);
				break;
			case 49:
				res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 2, 7, d_fin, multiplicativeForm);
				break;
			case 64:
				//calculateWeightCH2(bits, n, k, 6);
				res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 6, 2,  d_fin, multiplicativeForm);
				break;
			default:
				if (is_prime_power[q]) {
					res = count_number_of_wordsBytes_less_than(bitsCharCF, n, k, 1, q, d_fin, multiplicativeForm);
				}
				else {
					printf("not prime <64 \n\n");
				}
				break;
			}
		}
		else {
			//printf("Calculation using lookup tables!\n\n");
			write = fopen("Result_codewords_CountLessThan.txt","a");
			if (write!=NULL) {
				fprintf(write, "n = %d, k = %d, q = %d\n", n, k, q);
				fprintf(write, "Searching for words with weight < %d:\n", d_fin);
			}
			else {
				printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
			}
			linear_combinations_less_than_count(1, 1,d_fin);
			res = 0;
			for (int i = 1; i < d_fin; i++) {
				res = res + weights[i];
			}
			//res = res * (q - 1);
			fclose(write);
		}



	return res;
}



unsigned long long int count_word_equal_fixed_w( int d_fin, bool multiplicativeForm) {
#if FORCE_INSTR==0
	detect();
#else
	instructionSet = FORCE_INSTR;
#endif
	if (test >= 0) { instructionSet = test; }
	unsigned long long int res = 0;

	//if (!forceISTR) { detect(); }
	dispatch_try(instructionSet);
		if (instructionSet >= 5) { // blend SSE4.1
			switch (q)
			{
			case 2:
				res = count_number_of_wordsGF2_equal(bits, n, k, d_fin,multiplicativeForm);
				break;
			case 3:
				res = count_number_of_wordsCH3_equal(bits, n, k, 1, d_fin,multiplicativeForm);
				break;
			case 4:
				//calculateWeightBytes(bitsCharCF, n, k, 2, 2);
				res = count_number_of_wordsCH2_equal(bits, n, k, 2, d_fin,multiplicativeForm);
				break;
			case 8:
				res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 3, 2, d_fin,multiplicativeForm);
				//calculateWeightCH2(bits, n, k, 3);// bitwise representation can be used instead of byte representation
				break;
			case 9:
				res = count_number_of_wordsCH3_equal(bits, n, k, 2, d_fin,multiplicativeForm);
				break;
			case 16:
				res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 4, 2, d_fin,multiplicativeForm);
				//calculateWeightCH2(bits, n, k, 4);
				break;
			case 25:
				res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 2, 5, d_fin,multiplicativeForm);
				break;
			case 32:
				res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 5, 2, d_fin, multiplicativeForm);
				//calculateWeightCH2(bits, n, k, 5);
				break;
			case 27:
				res = count_number_of_wordsCH3_equal(bits, n, k, 3, d_fin,multiplicativeForm);
				break;
			case 49:
				res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 2, 7, d_fin,multiplicativeForm);
				break;
			case 64:
				//calculateWeightCH2(bits, n, k, 6);
				res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 6, 2, d_fin,multiplicativeForm);
				break;
			default:
				if (is_prime_power[q]) {
					res = count_number_of_wordsBytes_equal(bitsCharCF, n, k, 1, q, d_fin,multiplicativeForm);
				}
				else {
					printf("not prime <64 \n\n");
				}
				break;
			}
		}
		else {
			//printf("Calculation using lookup tables!\n\n");
			write = fopen("Result_codewords_CountEqual.txt", "a");
			if (write != NULL) {
				fprintf(write, "n = %d, k = %d, q = %d\n", n, k, q);
				fprintf(write, "Searching for words with weight = %d:\n", d_fin);
			}
			else {
				printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
			}

			linear_combinations_equal_count(1, 1, d_fin);

			fclose(write);
			res = 0;
			res = weights[d_fin];

		}


	return res;
}




// ----------------------- End user functions --------------------------//
void calculateWeightDistribution(char* name) {

	//fstream out;
	FILE* input = fopen(name, "r");
	if (input != NULL) {
		int num_of_input = 0;
		count_mat = 0;
		int error = 0;
		char ccc;// = getc(input);
		while (!(feof(input))) {
			ccc = getc(input);
			if (ccc == '?' || ccc == '!') {
				error = fscanf(input, "%d", &k);
				error = fscanf(input, "%d", &n);
				error = fscanf(input, "%d", &q);
				error = fscanf(input, "%d", &num_of_input);
				count_mat ++;
				if ((k > (n)) || (k < 2) || (n < 2)) {
					printf("Invalid n and / or k values!!\n");
					return;
				}
				if (!is_prime_power[q] && (q != 25) && (q != 49)) {
					printf("%d is not prime power!\n\n", q);
					return;
				}
				if (init_zero()) {
					matrix.k = k;
					matrix.n = n;
					matrix.q = q;
					matrix.num = num_of_input;
					bool read = readMatrix(input, n, k, q);
					char outputFile[] = "Weight_dis.txt";
					if (!read) {
						multipl = (ccc == '!');

						printMatrix(multipl, outputFile);
						if (init()) {
							clock_t begin_all, end_all;
							double time_all = 0;
							begin_all = clock();
							calculateWeightDistrib();
							end_all = clock();
							time_all = (end_all - begin_all) / (double(CLOCKS_PER_SEC));
							printf("time (%d, %d, %d) = %.2f\n", n, k, q, time_all);
						}
						else { printf("error init\n"); }

					}
					else {
						printf("Error in reading\n");
						return;
					}
					//printf("end of calc\n write weight dis\n");
					printWeights(weights, n, outputFile);



				}
				else {
					printf("Not enough memory!!!\n");
					fclose(input);
					return;
				}
			}
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		printf("Closing the file\n\n");
		fclose(input);
	}
	else { ERRORQ("Can't open input file\n"); }
}
void calculateWeightDistribution(int C, int R, int Q) {
	multipl = false;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return;
	}
	if (!is_prime_power[q] && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return;
	}
	if (init_zero()) {
		randomgenf(n, k, q,1);
		if (init()) {
			calculateWeightDistrib();
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
	}
	else {
		printf("Not enough memory!!!\n");
		return;
	}


}
void calculateWeightDistribution( int** input, int C, int R, int Q, bool multiplicativeForm) {
	if (multiplicativeForm && Q!= 4 && Q != 8 && Q != 16 && Q != 32 && Q != 64 && Q != 9 && Q != 27 && Q != 25 && Q != 49) {
		printf("Multiplicative form can be used only for composite fields!\n\n");
		return;
	}
	multipl = multiplicativeForm;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return;
	}
	if (!is_prime_power[q] && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return;
	}
	if (input == NULL) {
		printf("invalid input (NULL pointer)!");
		return;
	}
	if (init_zero()) {
		for (int i = 0; i < R; i++) {
			if (input[i] == NULL) {
				printf("invalid input (NULL pointer)!");
				return;
			}
			for (int j = 0; j < C; j++) {
				matrix.a[i + 1][j] = input[i][j]; // static matrix is indexed from [1][0]
			}
		}
		if (init()) {
			calculateWeightDistrib();
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}

	}
	else {
		printf("Not enough memory!!!\n");
		return;
	}


}


void find_word_less_than_fixed_weight(char* name, unsigned long long int w_searched) {

	FILE* input = fopen(name, "r");
	if (input != NULL) {
		int num_of_input = 0;
		int error = 0;
		count_mat = 0;
		char ccc;// = getc(input);
		FILE *out;
		//ofstream out;
		while (!(feof(input))) {
			ccc = getc(input);
			if (ccc == '?' || ccc == '!') {
				error = fscanf(input, "%d", &k);
				error = fscanf(input, "%d", &n);
				error = fscanf(input, "%d", &q);
				//fscanf(input, "%d", &d_fin);
				d_fin = w_searched;
				error = fscanf(input, "%d", &num_of_input);

				bool res = true;

				if ((k > (n)) || (k < 2) || (n < 2)) {
					printf("Invalid n and / or k values!!\n");
					return;
				}
				if (!is_prime_power[q] && (q != 25) && (q != 49)) {
					printf("%d is not prime power!\n\n", q);
					return;
				}

				if (init_zero()) {
					matrix.k = k;
					matrix.n = n;
					matrix.q = q;
					matrix.num = num_of_input;
					bool read = readMatrix(input, n, k, q);
					char outputFile[] = "find_less_than.txt";
					if (!read) {
						multipl = (ccc == '!');
						printMatrix(multipl, outputFile);
						if (init()) {
							res = find_word_less_than_fixed_w((int)d_fin);
						}
						else {
							res = false;
						}
						if (res) { //printMatrix(multipl, "find_less_than.txt");
						count_mat++;	}
						out = fopen("find_less_than.txt", "a");
						//out.open("find_less_than.txt", ios::app);
						if (res) {
							fprintf(out,"Found a word with weight less than %llu\n",d_fin);
							//out << "Found a word with weight less than " << d_fin << endl;
						}
						else {
							fprintf(out, "Did NOT find a word with weight less than %llu\n",d_fin);
							//out << "Did NOT find a word with weight less than " << d_fin << endl;
						}
						fclose(out);
						//out.close();
					}
					else {
						printf("Error in reading\n");
						return;
					}


				}
				else {
					printf("Not enough memory!!!\n");
					//cout << "Not enough memory!!!" << endl;
					fclose(input);
					return;
				}
			}
		}
		out = fopen("find_less_than.txt", "a");
		fprintf(out, "Total number = %llu\n", count_mat);
		fclose(out);
		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		printf("Closeing the file\n\n");
		fclose(input);
	}
	else {
		ERRORQ("Can't open input file\n");
	}
}
bool find_word_less_than_fixed_weight(int C, int R, int Q, int d_fin) {
	multipl = false;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return false;
	}
	if (!is_prime_power[q] && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return false;
	}
	if (init_zero()) {

		randomgenf(n, k, q,1);

		bool res;
		if (init()) {
			res = find_word_less_than_fixed_w(d_fin);
		}
		else {
			res = false;
		}


		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return res;
	}
	else {
		printf("Not enough memory!!!\n");
		return false;
	}

}
bool find_word_less_than_fixed_weight(int** input,int C, int R, int Q, int d_fin, bool multiplicativeForm) {
		if (multiplicativeForm && Q!= 4 && Q != 8 && Q != 16 && Q != 32 && Q != 64 && Q != 9 && Q != 27 && Q != 25 && Q != 49) {
			printf("Multiplicative form can be used only for composite fields!\n\n");
			return false;
	}
	multipl = multiplicativeForm;

	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return false;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return false;
	}
	if (input == NULL) {
		printf("invalid input (NULL pointer)!");
		return false;
	}
	if (init_zero()) {

		for (int i = 0; i < R; i++) {
			if (input[i] == NULL) {
				printf("invalid input (NULL pointer)!");
				return false;
			}
			for (int j = 0; j < C; j++) {
				matrix.a[i + 1][j] = input[i][j]; //  matrix is indexed from [1][0]
				matrix.a[i + 1][j] = input[i][j]; //  matrix is indexed from [1][0]
			}
		}
		bool res;
		if (init()) {
			res = find_word_less_than_fixed_w(d_fin);
		}
		else {
			res = false;
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}

		return res;
	}
	else {
		printf("Not enough memory!!!\n");
		return false;
	}
}



void find_word_equal_to_fixed_weight(char* name, unsigned long long int w_searched) {
	FILE* input = fopen(name, "r");
	if (input != NULL) {
		int num_of_input = 0;
		count_mat = 0;
		int error = 0;
		//ofstream out;
		FILE* out;
		char ccc;// = getc(input);
		while (!(feof(input))) {
			ccc = getc(input);
			if (ccc == '?' || ccc == '!') {
				error = fscanf(input, "%d", &k);
				error = fscanf(input, "%d", &n);
				error = fscanf(input, "%d", &q);
				//fscanf(input, "%d", &d_fin);
				d_fin = w_searched;
				error = fscanf(input, "%d", &num_of_input);
				//count_mat++;
				if ((k > (n)) || (k < 2) || (n < 2)) {
					printf("Invalid n and / or k values!!\n");
					return;
				}

				if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
					printf("%d is not prime power!\n\n", q);
					return;
				}
				if (init_zero()) {
					matrix.k = k;
					matrix.n = n;
					matrix.q = q;
					matrix.num = num_of_input;
					bool res = true;
					bool read = readMatrix(input, n, k, q);
					char outputFile[] = "find_equal.txt";
					if (!read) {
						multipl = (ccc == '!');
						printMatrix(multipl, outputFile);
						if (init()) {
							res = find_word_equal_fixed_w((int)d_fin);
						}
						else {
							res = false;
						}
						if (res) { //printMatrix(multipl,"find_equal.txt");
						count_mat++;
						};

						//out.open("find_equal.txt", ios::app);
						out = fopen("find_equal.txt", "a");
						if (res) {
							fprintf(out, "Found a word with weight equal to %llu\n", d_fin);
							//out << "Found a word with weight equal to " << d_fin << endl;
						}
						else {
							fprintf(out, "Did NOT find a word with weight equal to %llu\n", d_fin);
							//out << "Did NOT find a word with weight equal to" << d_fin << endl;
						}
						fclose(out);
						//out.close();
					}
					else {
						printf("Error in reading !\n");
						return;
					}


				}
				else {
					printf("Not enough memory!!\n");
					fclose(input);
					return;
				}
			}
		}
		out = fopen("find_equal.txt", "a");
		fprintf(out, "Total number = %llu", count_mat);
		fclose(out);
		//out.open("find_equal.txt", ios::app);
		//out << "Total number = " << count_mat << endl;
		//out.close();
		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}

		printf("Closeing the file\n\n");
		fclose(input);
	}
	else {
		ERRORQ("Can't open input file\n");
	}

}
bool find_word_equal_to_fixed_weight(int C, int R, int Q, int d_fin) {
	multipl = false;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return false;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return false;
	}
	if (init_zero()) {
		randomgenf(n, k, q,1);
		bool res;
		if (init()) {

			 res = find_word_equal_fixed_w(d_fin);
		}
		else {
			res = false;
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return res;

	}
	else {
		printf("Not enough memory\n");
		return false;
	}

}
bool find_word_equal_to_fixed_weight(int** input,int C, int R, int Q, int d_fin, bool multiplicativeForm) {
	if (multiplicativeForm && Q!= 4 && Q != 8 && Q != 16 && Q != 32 && Q != 64 && Q != 9 && Q != 27 && Q != 25 && Q != 49) {
		printf("Multiplicative form can be used only for composite fields!\n\n");
		return false;
	}
	multipl = multiplicativeForm;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return false;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return false;
	}
	if (input == NULL) {
		printf("invalid input (NULL pointer)!");
		return false;
	}
	if (init_zero()) {
		for (int i = 0; i < R; i++) {
			if (input[i] == NULL) {
				printf("invalid input (NULL pointer)!");
				return false;
			}
			for (int j = 0; j < C; j++) {
				matrix.a[i + 1][j] = input[i][j]; //  matrix is indexed from [1][0]
				matrix.a[i + 1][j] = input[i][j]; //  matrix is indexed from [1][0]
			}
		}
		bool res;
		if (init()) {
			res = find_word_equal_fixed_w(d_fin);
		}
		else {
			res = false;
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return res;
	}
	else {
		printf("Not enough memory\n");
		return false;
	}
}





void calculate_number_of_words_with_fixed_w(char* name, unsigned long long int w_searched,  bool write) {
	FILE* input = fopen(name, "r");
	if (input != NULL) {
		int num_of_input = 0;
		count_mat = 0;
		int error = 0;
		char ccc;// = getc(input);
		while (!(feof(input))) {
			ccc = getc(input);
			if (ccc == '?' || ccc == '!') {
				error = fscanf(input, "%d", &k);
				error = fscanf(input, "%d", &n);
				error = fscanf(input, "%d", &q);
				//fscanf(input, "%d", &d_fin);
				d_fin = w_searched;
				error = fscanf(input, "%d", &num_of_input);
				count_mat ++;
				if ((k > (n)) || (k < 2) || (n < 2)) {
					printf("Invalid n and / or k values!!\n");
					return;
				}
				if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
					printf("%d is not prime power!\n\n", q);
					return;
				}

				if (init_zero()) {
					matrix.k = k;
					matrix.n = n;
					matrix.q = q;
					matrix.num = num_of_input;
					unsigned long long int ct = 0;
					bool read = readMatrix(input, n, k, q);
					char outputFile[] = "count_equal.txt";
					if (!read) {
						multipl = (ccc == '!');
						printMatrix(multipl, outputFile);
						if (init()) {
							if (write) {
								ct = count_word_equal_fixed_w((int)d_fin, multipl) * (q - 1);
							}
							else {
								calculateWeightDistrib();
								ct = weights[d_fin];
							}
						}
						else {
							ct = 0;
						}
						FILE* out;
						out = fopen("count_equal.txt", "a");
						fprintf(out, "Found %llu words with weight equal to %llu\n\n\n", ct, d_fin);
						fclose(out);

					}
					else {
						printf("Error in reading !\n");
						return;
					}



				}
				else {
					printf("Not enough memory\n");
					fclose(input);
					return;
				}
			}
		}
		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		printf("Closeing the file\n\n");
		fclose(input);
	}
	else {
		ERRORQ("Can't open input file\n");
	}

}
unsigned long long int calculate_number_of_words_with_fixed_w(int C, int R, int Q, int w_searched, bool write) {
	multipl = false;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return 0;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return 0;
	}

	if (init_zero()) {
		unsigned long long int ct = 0;
		randomgenf(n, k, q,1);
		if (init()) {
			if (write) {
				ct = count_word_equal_fixed_w(w_searched, multipl) * (Q - 1);
			}
			else {
				calculateWeightDistrib();
				ct = weights[w_searched];
			}
		}
		else { ct = 0; }

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return ct;
	}
	else {
		printf("Not enough memory\n");
		return 0;
	}


}
unsigned long long int calculate_number_of_words_with_fixed_w(int** input,int C, int R, int Q, int w_searched, bool write, bool multiplicativeForm) {
	if (multiplicativeForm && Q!= 4 && Q != 8 && Q != 16 && Q != 32 && Q != 64 && Q != 9 && Q != 27 && Q != 25 && Q != 49) {
		printf("Multiplicative form can be used only for composite fields!\n\n");
		return 0;
	}
	multipl = multiplicativeForm;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return 0;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return 0;
	}
	if (input == NULL) {
		printf("invalid input (NULL pointer)!");
		return 0;
	}
	unsigned long long int ct = 0;
	if (init_zero()) {
		for (int i = 0; i < R; i++) {
			if (input[i] == NULL) {
				printf("invalid input (NULL pointer)!");
				return 0;
			}
			for (int j = 0; j < C; j++) {
				matrix.a[i + 1][j] = input[i][j]; //  matrix is indexed from [1][0]
			}
		}
		if (init()) {
			if (write) {
				ct = count_word_equal_fixed_w(w_searched,multipl) * (Q - 1);
			}
			else {
				calculateWeightDistrib();
				ct = weights[w_searched];
			}
		}
		else { ct = 0; }

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return ct;
	}
	else {
		printf("Not enough memory\n");
		return 0;
	}
}


void calculate_number_of_words_less_than_fixed_w(char* name, unsigned long long int w_searched,  bool write) {
	FILE* input = fopen(name, "r");
	if (input != NULL) {
		int num_of_input = 0;
		count_mat = 0;
		int error = 0;
		char ccc;// = getc(input);
		while (!(feof(input))) {
			ccc = getc(input);
			if (ccc == '?' || ccc == '!') {
				error = fscanf(input, "%d", &k);
				error = fscanf(input, "%d", &n);
				error = fscanf(input, "%d", &q);
				//fscanf(input, "%d", & d_fin);
				d_fin = w_searched;
				error = fscanf(input, "%d", &num_of_input);
				count_mat++;

				if ((k > (n)) || (k < 2) || (n < 2)) {
					printf("Invalid n and / or k values!!\n");
					return;
				}
				if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
					printf("%d is not prime power!\n\n", q);
					return;
				}
				if (init_zero()) {
					matrix.k = k;
					matrix.n = n;
					matrix.q = q;
					matrix.num = num_of_input;
					unsigned long long int count = 0;
					bool read = readMatrix(input, n, k, q);
					char outputFile[] = "count_less_than.txt";
					if (!read) {
						multipl = (ccc == '!');
						printMatrix(multipl, outputFile);
						if (init()) {
							if (write) {
								count = count_word_less_than_fixed_w((int)d_fin, multipl) * (q - 1);
							}
							else {
								calculateWeightDistrib();
								for (int i = 0; i < d_fin; i++) {
									count = count + weights[i];
								}
							}
						}
						else {
							count = 0;
						}
						FILE* out = fopen("count_less_than.txt", "a");
						fprintf(out, " Found %llu words with weight less than %llu \n", count, d_fin);
						fclose(out);

						//ofstream out;
						//out.open("count_less_than.txt", ios::app);
						//out << " Found  " << count << " words with weight less than " << d_fin << endl;
						//out.close();

					}
					else {
						printf("Error in reading !\n");
						return;
					}


				}
				else {
					printf("Not enough memory\n");
					fclose(input);
					return;
				}
			}
		}
		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		printf("Closeing the file\n\n");
		fclose(input);
	}
	else {
		ERRORQ("Can't open input file\n");
	}

}
unsigned long long int calculate_number_of_words_less_than_fixed_w(  int C, int R, int Q, int w_searched, bool write) {
	multipl = false;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return 0;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return 0;
	}
	if (init_zero()) {
		unsigned long long int count = 0;
		randomgenf(n, k, q,1);
		if (init()) {
			if (write) {
				count = count_word_less_than_fixed_w(w_searched, multipl) * (Q - 1);
			}
			else {
				calculateWeightDistrib();
				for (int i = 0; i < w_searched; i++) {
					count = count + weights[i];
				}
			}
		}
		else {
			count = 0;
		}
		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return count;
	}
	else {
		printf("Not enough memory\n");
		return 0;
	}


}
unsigned long long int calculate_number_of_words_less_than_fixed_w(int** input,int C, int R, int Q,   int w_searched, bool write, bool multiplicativeForm) {
	if (multiplicativeForm && Q!= 4 && Q != 8 && Q != 16 && Q != 32 && Q != 64 && Q != 9 && Q != 27 && Q != 25 && Q != 49) {
		printf("Multiplicative form can be used only for composite fields!\n");
		return 0;
	}
	multipl = multiplicativeForm;
	n = C; k = R; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return 0;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return 0;
	}
	if (input == NULL) {
		printf("invalid input (NULL pointer)!");
		return 0;
	}
	if (init_zero()) {
		for (int i = 0; i < R; i++) {
			if (input[i] == NULL) {
				printf("invalid input (NULL pointer)!");
				return 0;
			}
			for (int j = 0; j < C; j++) {
				matrix.a[i + 1][j] = input[i][j]; // static matrix is indexed from [1][0]
			}
		}
		unsigned long long int count = 0;
		if (init()) {
			if (write) {
				count = count_word_less_than_fixed_w(w_searched,multipl) * (Q - 1);
			}
			else {
				calculateWeightDistrib();
				for (int i = 0; i < w_searched; i++) {
					count = count + weights[i];
				}
			}
		}
		else {
			count = 0;
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return count;
	}
	else {
		printf("Not enough memory\n");
		return 0;
	}

}

void min_dis(char* name ) {

	FILE* input = fopen(name, "r");
	if (input != NULL) {
		int num_of_input = 0;
		count_mat = 0;
		int error = 0;
		char ccc;// = getc(input);
		while(!(feof(input))) {
			ccc = getc(input);
			if (ccc == '?' || ccc == '!') {
				error = fscanf(input, "%d", &k);
				error = fscanf(input, "%d", &n);
				error = fscanf(input, "%d", &q);
				error = fscanf(input, "%d", &num_of_input);
				count_mat++;
				if ((k > (n)) || (k < 2) || (n < 2)) {
					printf("Invalid n and / or k values!!\n");
					return;
				}
				if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
					printf("%d is not prime power!\n\n", q);
					return;
				}
				if (init_zero()) {
					matrix.k = k;
					matrix.n = n;
					matrix.q = q;
					matrix.num = num_of_input;
					int i = 0;
					bool read = readMatrix(input, n, k, q);
					char outputFile[] = "min_distance.txt";
					if (!read) {
						bool read = (ccc == '!');
						printMatrix(multipl, outputFile);
						if (init()) {
							calculateWeightDistrib();
							//find min d
							i++;
							while (weights[i] == 0) {
								i++;
							}
						}
						else {
							i = 0;
						}
						FILE* out = fopen("min_distance.txt", "a");
						fprintf(out, "d = %d\n", i);
						fclose(out);

						//ofstream out;
						//out.open("min_distance.txt", ios::app);
						//out << "d = " << i << endl;
						//out.close();

					}
					else {
						printf("Error in reading !\n");
						return;
					}
				}
				else {
					printf("Not enough memory\n");
					fclose(input);
					return;
				}
			}
		}
		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		printf("Closeing the file\n\n");
		fclose(input);
	}
	else {
		ERRORQ("Can't open input file\n");
	}


}
unsigned long long int min_dis(  int N, int K, int Q) {
	int num = 1;
	multipl = false;
	n = N; k = K; q = Q;
	if ((k > (n)) || (k < 2) || (n < 2)) {
		printf("Invalid n and / or k values!!\n");
		return 0;
	}
	if ((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return 0;
	}
	if (init_zero()) {

		randomgenf(n, k, q, num);

		int i = 1;
		if (init()) {
			calculateWeightDistrib();

			//find min d
			while (weights[i] == 0) {
				i++;
			}
		}
		else {
			i = 0;
		}


		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return i;
	}
	else {
		printf("Not enough memory\n");
		return 0;
	}

}
unsigned long long int min_dis(int** input, int N, int K, int Q, bool multiplicativeForm) {
	if (multiplicativeForm && Q!= 4 && Q != 8 && Q != 16 && Q != 32 && Q != 64 && Q != 9 && Q != 27 && Q != 25 && Q != 49) {
		printf("Multiplicative form can be used only for composite fields!\n\n");
		return 0;
	}
	multipl = multiplicativeForm;
	n = N; k = K; q = Q;
	if ((k > (n))||(k<2)||(n<2)) {
		printf( "Invalid n and / or k values!!\n\n" );
		return 0;
	}
	if((!is_prime_power[q]) && (q != 25) && (q != 49)) {
		printf("%d is not prime power!\n\n",q);
		return 0;
	}
	if (input == NULL) {
		printf("invalid input (NULL pointer)!");
		return 0;
	}
	if (init_zero()) {
		for (int i = 0; i < K; i++) {
			if (input[i] == NULL) {
				printf("invalid input (NULL pointer)!");
				return 0;
			}
			for (int j = 0; j < N; j++) {
				matrix.a[i + 1][j] = input[i][j]; //  matrix is indexed from [1][0]
			}
		}

		int i = 1;
		if (init()) {
			calculateWeightDistrib();

			while (weights[i] == 0) {
				i++;
			}
		}
		else {
			i = 0;
		}

		dmat_short_free(matrix);
		dmat_short_free(matrixH);
		if (bits.mem != 0) {
			dmat_short_free(bits);
		}
		if (bitsCharCF.mem != 0) {
			dmat_free(bitsCharCF);
		}
		return i;
	}
	else {
		printf("Not enough memory\n");
		return 0;
	}

}
